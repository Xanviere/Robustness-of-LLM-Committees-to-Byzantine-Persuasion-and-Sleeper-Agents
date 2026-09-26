import asyncio
import argparse
import json
import os
import sys
import math
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.client_wrapper import LLMClient
from core.concurrency import ConcurrencyManager
from debate.engine import DebateEngine
from attacks.redteam_agent import ByzantineAgent
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

async def run_stage1():
    parser = argparse.ArgumentParser(description="Stage 1: Symmetric Model-Homogeneous Pilot Sweep")
    parser.add_argument("--manifest", type=str, default="data/manifests/stage1_frontier_hard_manifest.jsonl", help="Path to input manifest")
    parser.add_argument("--model", type=str, default="gemini-3.7-flash", help="Primary model identifier")
    parser.add_argument("--provider", type=str, default="gemini", help="API Provider (gemini, openai, etc)")
    parser.add_argument("--replicates", type=int, default=3, help="Stochastic replicates K (default: 3)")
    parser.add_argument("--rounds", type=int, default=2, help="Debate rounds R (default: 2)")
    parser.add_argument("--output-log", type=str, default="data/raw_logs/stage1_pilot.jsonl", help="Telemetry log output path")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests (default: 8)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for honest agents (default: 0.7)")
    
    args = parser.parse_args()

    api_key = os.environ.get(f"{args.provider.upper()}_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print(f"\n[WARNING] {args.provider.upper()}_API_KEY is not set. The client wrapper might try to load it internally or fail.")
        pass # Removed hard error
        pass # Removed hard error instructions
        return

    if not os.path.exists(args.manifest):
        print(f"[ERROR] Manifest file {args.manifest} not found.")
        return

    with open(args.manifest, "r", encoding="utf-8-sig") as f:
        items = [json.loads(line.strip()) for line in f if line.strip()]

    if os.path.exists(args.output_log):
        try:
            os.remove(args.output_log)
        except Exception:
            pass

    # Stage 1 Experimental Grid:
    # N in {3, 5} x Threats {Blind Post-Anchor, Oracle Post-Anchor} x K Replicates
    committee_sizes = [3, 5]
    threat_conditions = [
        {
            "id": "blind_postanchor_adaptive",
            "knowledge": "blind",
            "onset": "post_anchor",
            "adaptivity": "round_adaptive",
            "objective": "targeted_error",
            "influence_channels": {"forced_confidence": True, "spurious_proof": True, "authoritative_rhetoric": True}
        },
        {
            "id": "oracle_postanchor_adaptive",
            "knowledge": "oracle",
            "onset": "post_anchor",
            "adaptivity": "round_adaptive",
            "objective": "targeted_error",
            "influence_channels": {"forced_confidence": True, "spurious_proof": True, "authoritative_rhetoric": True}
        }
    ]

    total_conditions = len(committee_sizes) * len(threat_conditions)
    total_trajectories = total_conditions * len(items) * args.replicates

    print(f"\n==================================================================")
    print(f" STAGE 1: SYMMETRIC MODEL-HOMOGENEOUS PILOT SWEEP")
    print(f" Model               : {args.model}")
    print(f" Committee Sizes (N) : {committee_sizes}")
    print(f" Debate Rounds (R)   : {args.rounds}")
    print(f" Stochastic Reps (K) : {args.replicates}")
    print(f" Threats Evaluated   : {[t['id'] for t in threat_conditions]}")
    print(f" Benchmark Items     : {len(items)} items")
    print(f" Total Trajectories  : {total_trajectories} paired runs")
    print(f" Concurrency Limit   : {args.concurrency}")
    print(f" Output Telemetry    : {args.output_log}")
    print(f"==================================================================\n")

    concurrency_mgr = ConcurrencyManager(
        max_concurrent=args.concurrency,
        max_retries=4,
        hard_spend_ceiling_usd=750.0,
        initial_backoff_seconds=1.5
    )

    all_results = []
    condition_summaries = {}

    for N in committee_sizes:
        clients = [
            LLMClient(provider=args.provider, model_name=args.model, api_key=api_key)
            for _ in range(N)
        ]

        engine = DebateEngine(
            clients=clients,
            concurrency_manager=concurrency_mgr,
            study_id="stage1_symmetric_pilot",
            raw_log_path=args.output_log
        )

        for threat in threat_conditions:
            cond_key = f"N{N}_{threat['id']}"
            print(f"\n>>> Running Condition: Committee N={N} | Threat: {threat['id']} <<<")

            attacker = ByzantineAgent(
                knowledge_set=threat["knowledge"],
                onset_timing=threat["onset"],
                adaptivity=threat["adaptivity"],
                objective=threat["objective"],
                influence_channels=threat["influence_channels"]
            )

            cond_results = []

            for rep in range(args.replicates):
                for idx, it in enumerate(items):
                    item_id = it.get("item_id", f"item_{idx}")
                    prompt = it.get("prompt", "")
                    gt = it.get("ground_truth", "")
                    dom = it.get("domain", "aime")
                    plausible_wrong = it.get("plausible_wrong", None)

                    print(f"  --> [Rep {rep+1}/{args.replicates}] Item [{idx+1}/{len(items)}]: {item_id} (Domain: {dom})")

                    try:
                        res = await engine.run_matched_trajectory(
                            item_id=item_id,
                            prompt=prompt,
                            ground_truth=gt,
                            domain=dom,
                            replicate_id=rep,
                            R=args.rounds,
                            byzantine_agent=attacker,
                            threat_config=threat,
                            active_defense_name="none",
                            oracle_wrong_target=plausible_wrong,
                            temperature=args.temperature
                        )
                        cond_results.append(res)
                        all_results.append(res)

                    except Exception as e:
                        logger.error(f"Error on {item_id} (Rep {rep}): {e}")

            # Analyze this specific condition
            t1 = analyze_tier1_operational(cond_results)
            t2 = analyze_tier2_mechanism(cond_results)

            condition_summaries[cond_key] = {
                "N": N,
                "threat": threat["id"],
                "total_runs": len(cond_results),
                "clean_acc": t1.get("clean_accuracy", 0.0),
                "attacked_acc": t1.get("attacked_accuracy", 0.0),
                "s_majority_size": t2.get("conditioned_population_size", 0),
                "s_majority_yield": (t2.get("conditioned_population_size", 0) / max(1, len(cond_results))),
                "AAE": t2.get("AAE", 0.0),
                "VSR": t2.get("VSR", 0.0),
                "CCR": t2.get("CCR", 0.0),
                "p_harmful_revision": t2.get("p_harmful_revision_given_r0_correct", 0.0)
            }

    # ----------------------------------------------------
    # FINAL STAGE 1 PILOT REPORT & POWER ANALYSIS
    # ----------------------------------------------------
    tracker = CostTracker()
    tracker.load_telemetry_file(args.output_log)
    c_summary = tracker.compute_summary()

    print("\n==================================================================")
    print(" STAGE 1 PILOT SWEEP COMPLETE - SUMMARY & POWER TARGETS")
    print("==================================================================")
    print(f"Total API Calls Emitted : {c_summary['total_calls']}")
    print(f"Total Input Tokens      : {c_summary['total_input_tokens']}")
    print(f"Total Output Tokens     : {c_summary['total_output_tokens']}")
    print(f"Total Measured Spend    : ${c_summary['total_cost_usd']:.4f} USD")
    print(f"Mean Latency Per Call   : {c_summary['mean_latency_seconds']:.3f} s\n")

    summary_df = pd.DataFrame.from_dict(condition_summaries, orient="index")
    print("--- CONDITION-BY-CONDITION PERFORMANCE ---")
    print(summary_df[["N", "threat", "clean_acc", "attacked_acc", "s_majority_yield", "AAE", "VSR", "CCR"]])

    # Statistical Power Calculation for Stage 2 (Confirmatory Target)
    # Target power 1 - beta = 0.80, alpha = 0.05
    # z_{1 - alpha/2} = 1.96, z_{1 - beta} = 0.842
    z_alpha = 1.96
    z_beta = 0.842
    
    # Calculate pooled empirical delta and variance across eligible pairs
    deltas = []
    for r in all_results:
        c_verdict = r.get("clean_consensus", {}).get("verdict", "")
        a_verdict = r.get("attacked_consensus", {}).get("verdict", "")
        gt = r.get("ground_truth", "")
        dom = r.get("domain", "aime")
        
        c_corr = (str(c_verdict).strip().lower() == str(gt).strip().lower())
        a_corr = (str(a_verdict).strip().lower() == str(gt).strip().lower())
        
        if c_corr and not a_corr:
            deltas.append(1.0)
        elif not c_corr and a_corr:
            deltas.append(-1.0)
        else:
            deltas.append(0.0)

    delta_arr = np.array(deltas)
    emp_mean_aae = float(np.mean(delta_arr)) if len(delta_arr) > 0 else 0.0
    emp_var = float(np.var(delta_arr, ddof=1)) if len(delta_arr) > 1 else 0.25

    # Overall strict-majority eligibility rate p_hat
    total_pairs = len(all_results)
    total_eligible = sum(cond["s_majority_size"] for cond in condition_summaries.values())
    p_hat_eligible = total_eligible / max(1, total_pairs)

    # Minimum detectable effect size (MDE) target = 0.10 (or empirical mean if > 0.05)
    target_effect = max(0.10, abs(emp_mean_aae))
    required_n_pairs = ((z_alpha + z_beta) ** 2 * emp_var) / (target_effect ** 2)
    required_Q_items = math.ceil(required_n_pairs / (max(0.20, p_hat_eligible) * args.replicates))

    print("\n--- STATISTICAL POWER & STAGE 2 SAMPLE SIZE (Q) TARGET ---")
    print(f"Empirical Strict Majority Yield (p_hat) : {p_hat_eligible:.2%}")
    print(f"Empirical Effect Variance (sigma^2)    : {emp_var:.4f}")
    print(f"Target Effect Size Delta (MDE)         : {target_effect:.4f}")
    print(f"Required Matched Pairs (1-beta=0.80)   : {math.ceil(required_n_pairs)} pairs")
    print(f"Recommended Stage 2 Sample Size (Q)    : {required_Q_items} valid benchmark items/domain")
    print("==================================================================\n")

if __name__ == "__main__":
    asyncio.run(run_stage1())
