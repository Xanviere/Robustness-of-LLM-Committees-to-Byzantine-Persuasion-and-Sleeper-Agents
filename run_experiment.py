import asyncio
import argparse
import json
import os
import sys
import yaml
import logging
from typing import Dict, Any, List

# Safe stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.client_wrapper import LLMClient
from core.concurrency import ConcurrencyManager
from debate.engine import DebateEngine
from attacks.redteam_agent import ByzantineAgent
from attacks.sleeper import SleeperAgentTracker
from defenses.audit_anchor import AuditAnchor
from defenses.bayesian_filter import BayesianTrustFilter
from defenses.llm_judge import LLMJudgeVerifier
from evaluation.cost_tracker import CostTracker
from evaluation.tier1_operational import analyze_tier1_operational
from evaluation.tier2_mechanism import analyze_tier2_mechanism
from evaluation.statistical_tests import (
    question_clustered_bootstrap,
    permutation_test,
    benjamini_hochberg_adjustment
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

async def run_experiment_cli():
    parser = argparse.ArgumentParser(description="Byzantine MAS Benchmark & Protocol Evaluation Harness")
    parser.add_argument("--config", type=str, default="config/base_config.yaml", help="Path to base config YAML")
    parser.add_argument("--threat-config", type=str, default="config/threat_profiles.yaml", help="Path to threat profiles YAML")
    parser.add_argument("--defense-config", type=str, default="config/defense_profiles.yaml", help="Path to defense profiles YAML")
    parser.add_argument("--manifest", type=str, default="data/manifests/aime_test.jsonl", help="Path to input JSONL benchmark manifest")
    parser.add_argument("--threat-id", type=str, default="blind_postanchor_adaptive", help="Threat profile ID")
    parser.add_argument("--defense-id", type=str, default="none", help="Defense profile ID")
    parser.add_argument("--provider", type=str, default="gemini", help="LLM Provider (gemini, openai, anthropic, mock)")
    parser.add_argument("--model", type=str, default="gemini-3.7-flash", help="Model identifier")
    parser.add_argument("--committee-size", type=int, default=5, help="Committee size N in {3, 5, 7, 9}")
    parser.add_argument("--rounds", type=int, default=2, help="Debate rounds R")
    parser.add_argument("--replicates", type=int, default=1, help="Stochastic replicates per item")
    parser.add_argument("--stage", type=str, default="stage0", help="Stage 0, 1, 2, 3, 4A, 4B, 4C")
    parser.add_argument("--output-log", type=str, default="data/raw_logs/experiment_run.jsonl", help="Path for telemetry output")
    
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    threat_cfg = load_yaml(args.threat_config)
    defense_cfg = load_yaml(args.defense_config)

    # Resolve threat profile
    selected_threat = None
    for prof in threat_cfg.get("threat_profiles", []):
        if prof.get("id") == args.threat_id:
            selected_threat = prof
            break
    if not selected_threat:
        selected_threat = {
            "knowledge": "blind",
            "onset": "post_anchor",
            "adaptivity": "round_adaptive",
            "objective": "targeted_error",
            "influence_channels": {"forced_confidence": True, "spurious_proof": True, "authoritative_rhetoric": True}
        }

    # Resolve defense profile
    selected_defense = None
    for prof in defense_cfg.get("defense_profiles", []):
        if prof.get("id") == args.defense_id:
            selected_defense = prof
            break
    if not selected_defense:
        selected_defense = {"audit_anchor": False, "prompt_reminder": False, "verifier_gate": False, "hard_lock": False, "bayesian_filter": False}

    spend_ceiling = base_cfg.get("spend_caps", {}).get("hard_ceiling_usd", 750.0)
    concurrency_mgr = ConcurrencyManager(
        max_concurrent=base_cfg.get("concurrency", {}).get("max_concurrent", 10),
        max_retries=base_cfg.get("concurrency", {}).get("max_retries", 3),
        hard_spend_ceiling_usd=spend_ceiling
    )

    # Heterogeneous or Homogeneous Committee Setup
    if args.stage == "4B" or args.provider == "heterogeneous":
        model_panel = base_cfg.get("models", {}).get("heterogeneous_panel", ["gemini-3.7-flash", "gpt-5.6-luna", "claude-haiku-4.5"])
        clients = []
        for i in range(args.committee_size):
            m = model_panel[i % len(model_panel)]
            prov = "gemini" if "gemini" in m else ("openai" if "gpt" in m else "anthropic")
            clients.append(LLMClient(provider=prov, model_name=m))
    else:
        clients = [
            LLMClient(provider=args.provider, model_name=args.model)
            for _ in range(args.committee_size)
        ]

    engine = DebateEngine(
        clients=clients,
        concurrency_manager=concurrency_mgr,
        study_id=f"exp_{args.stage}_{args.threat_id}_{args.defense_id}",
        raw_log_path=args.output_log
    )

    audit_defense = AuditAnchor() if selected_defense.get("audit_anchor") else None
    bayesian_filter = None
    if selected_defense.get("bayesian_filter"):
        b_params = selected_defense.get("bayesian_params", {})
        bayesian_filter = BayesianTrustFilter(
            a0=b_params.get("a0", 1.0),
            b0=b_params.get("b0", 1.0),
            discount=b_params.get("discount_lambda", 0.85),
            threshold=b_params.get("quarantine_threshold_tau", 0.35),
            min_obs=b_params.get("min_observations", 3)
        )

    sleeper_tracker = None
    if selected_threat.get("onset") == "sleeper":
        sleeper_tracker = SleeperAgentTracker(K=selected_threat.get("sleeper_K", 5))

    attacker = ByzantineAgent(
        knowledge_set=selected_threat.get("knowledge", "oracle"),
        onset_timing=selected_threat.get("onset", "post_anchor"),
        adaptivity=selected_threat.get("adaptivity", "round_adaptive"),
        objective=selected_threat.get("objective", "targeted_error"),
        influence_channels=selected_threat.get("influence_channels", {}),
        sleeper_tracker=sleeper_tracker
    )

    # Role configuration (Phase 2 role sweeps)
    roles = None
    if args.stage == "4C" or "roles" in args.stage.lower():
        role_cycle = ["proposer", "critic", "verifier"]
        roles = [role_cycle[i % len(role_cycle)] for i in range(args.committee_size)]

    print(f"\n=======================================================")
    print(f" BYZANTINE MAS EXPERIMENTAL BENCHMARK HARNESS")
    print(f" Stage: {args.stage} | Model: {args.model} | N={args.committee_size} | R={args.rounds}")
    print(f" Threat Profile : {args.threat_id}")
    print(f" Defense Profile: {args.defense_id}")
    print(f" Manifest       : {args.manifest}")
    print(f" Telemetry Log  : {args.output_log}")
    print(f"=======================================================\n")

    if not os.path.exists(args.manifest):
        print(f"Error: Manifest {args.manifest} not found.")
        return

    with open(args.manifest, "r", encoding="utf-8-sig") as f:
        items = [json.loads(line.strip()) for line in f if line.strip()]

    trajectory_results = []

    for rep in range(args.replicates):
        for idx, item in enumerate(items):
            item_id = item.get("item_id", f"item_{idx}")
            prompt = item.get("prompt", "")
            gt = item.get("ground_truth", "")
            dom = item.get("domain", "aime")

            print(f"--> [Rep {rep+1}/{args.replicates}] Processing [{idx+1}/{len(items)}]: {item_id}")

            if sleeper_tracker:
                sleeper_tracker.advance_task()

            try:
                res = await engine.run_matched_trajectory(
                    item_id=item_id,
                    prompt=prompt,
                    ground_truth=gt,
                    domain=dom,
                    replicate_id=rep,
                    R=args.rounds,
                    byzantine_agent=attacker,
                    audit_anchor_defense=audit_defense,
                    bayesian_filter=bayesian_filter,
                    enable_prompt_reminder=selected_defense.get("prompt_reminder", False),
                    enable_verifier_gate=selected_defense.get("verifier_gate", False),
                    enable_hard_lock=selected_defense.get("hard_lock", False),
                    threat_config=selected_threat,
                    active_defense_name=args.defense_id,
                    roles=roles
                )
                trajectory_results.append(res)

                if bayesian_filter:
                    for agent_id, a_res in res["attacked_branch"].items():
                        ans = a_res.get("payload", {}).get("final_answer", "")
                        is_corr = 1.0 if (str(ans).strip().lower() == str(gt).strip().lower()) else 0.0
                        bayesian_filter.update_trust(agent_id, is_corr)

            except Exception as e:
                logger.error(f"Error processing item {item_id}: {e}")

    cost_tracker = CostTracker()
    cost_tracker.load_telemetry_file(args.output_log)
    c_summary = cost_tracker.compute_summary()

    t1 = analyze_tier1_operational(trajectory_results)
    t2 = analyze_tier2_mechanism(trajectory_results)

    print("\n=======================================================")
    print(" EXPERIMENT COMPLETE - FINAL MULTI-TIER REPORT")
    print("=======================================================")
    print(f"Total API Calls        : {c_summary['total_calls']}")
    print(f"Total Measured Spend   : ${c_summary['total_cost_usd']:.4f} USD")
    print(f"Mean Latency Per Call  : {c_summary['mean_latency_seconds']:.3f} s")
    print("\n--- TIER 1: FULL POPULATION OPERATIONAL METRICS ---")
    print(f"Clean Accuracy         : {t1.get('clean_accuracy', 0.0):.2%}")
    print(f"Attacked Accuracy      : {t1.get('attacked_accuracy', 0.0):.2%}")
    print(f"Consensus Liveness (C) : {t1.get('clean_consensus_liveness', 0.0):.2%}")
    print(f"Consensus Liveness (A) : {t1.get('attacked_consensus_liveness', 0.0):.2%}")
    print(f"Tie Rate (Clean/Atk)   : {t1.get('clean_tie_rate', 0.0):.2%} / {t1.get('attacked_tie_rate', 0.0):.2%}")
    print(f"Unconditional Risk Diff: {t1.get('unconditional_risk_difference', 0.0)}")

    print("\n--- TIER 2: STRICT MAJORITY CAUSAL MECHANISM ANALYSIS ---")
    print(f"Conditioned Set Size   : {t2.get('conditioned_population_size', 0)}")
    print(f"Split-Anchor Exclusions: {t2.get('split_anchor_count', 0)}")
    print(f"Average Attack Effect  : {t2.get('AAE', 0.0)} (AAE = VSR - CCR)")
    print(f"Vulnerability Rate VSR : {t2.get('VSR', 0.0)}")
    print(f"Clean Collapse Rate CCR: {t2.get('CCR', 0.0)}")
    print(f"Attacked Error Rate AER: {t2.get('AER', 0.0)}")
    print(f"P(Harmful Rev | R0 Cor): {t2.get('p_harmful_revision_given_r0_correct', 0.0):.2%}")
    print(f"P(Correct Rev | R0 Wrn): {t2.get('p_corrective_revision_given_r0_wrong', 0.0):.2%}")

    print("\nMatched 2x2 Contingency Table:")
    if "contingency_matrix_df" in t2:
        print(t2["contingency_matrix_df"])

    print("\nHonest Agent 5-State Transition Dynamics:")
    for state, cnt in t2.get("transition_dynamics", {}).items():
        print(f"  - {state.replace('_', ' ').title()}: {cnt}")
    print("=======================================================\n")

if __name__ == "__main__":
    asyncio.run(run_experiment_cli())
