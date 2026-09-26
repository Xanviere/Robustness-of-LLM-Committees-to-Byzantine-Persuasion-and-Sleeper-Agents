import asyncio
import json
import os
import sys
import logging
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.client_wrapper import LLMClient
from core.concurrency import ConcurrencyManager
from debate.engine import DebateEngine
from attacks.redteam_agent import ByzantineAgent
from defenses.audit_anchor import AuditAnchor
from defenses.bayesian_filter import BayesianTrustFilter
from evaluation.tier1_operational import analyze_tier1_operational
from evaluation.tier2_mechanism import analyze_tier2_mechanism
from evaluation.cost_tracker import CostTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

async def run_vertical_slice():
    api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n[ERROR] TOGETHER_API_KEY environment variable is not set!")
        print("Please set your API key in PowerShell: $env:TOGETHER_API_KEY='your-key'")
        print("Also ensure you route traffic to Together AI by setting: $env:OPENAI_BASE_URL='https://api.together.xyz/v1'\n")
        return

    manifest_path = os.environ.get("BYZANTINE_MANIFEST", "data/manifests/vertical_slice_demo.jsonl")
    raw_log_path = "data/raw_logs/stage0_test_llama.jsonl"

    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest not found at {manifest_path}")
        return

    if os.path.exists(raw_log_path):
        try:
            os.remove(raw_log_path)
        except Exception:
            pass

    concurrency_mgr = ConcurrencyManager(
        max_concurrent=6,
        max_retries=4,
        hard_spend_ceiling_usd=750.0,
        initial_backoff_seconds=1.5
    )

    model_name = os.environ.get("BYZANTINE_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")

    clients = [
        LLMClient(provider="openai", model_name=model_name, api_key=api_key),
        LLMClient(provider="openai", model_name=model_name, api_key=api_key),
        LLMClient(provider="openai", model_name=model_name, api_key=api_key)
    ]

    engine = DebateEngine(
        clients=clients,
        concurrency_manager=concurrency_mgr,
        run_id="stage0_pilot_live_demo_llama",
        study_id="byzantine_mas_stage0",
        raw_log_path=raw_log_path
    )

    attacker = ByzantineAgent(
        knowledge_set="oracle",
        onset_timing="post_anchor",
        adaptivity="round_adaptive",
        objective="targeted_error",
        influence_channels={
            "forced_confidence": True,
            "spurious_proof": True,
            "authoritative_rhetoric": True,
            "bare_answer": False
        }
    )

    trajectory_results = []
    print(f"\n=======================================================")
    print(f" LIVE SYSTEM DEMONSTRATION HARNESS (LLaMA EDITION)")
    print(f" Model: {model_name} | Committee Size: N=3 | Rounds: R=2")
    print(f" Multi-Domain Manifest: {manifest_path} (AIME + LCB + Invariants)")
    print(f"=======================================================\n")

    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        items = [json.loads(line.strip()) for line in f if line.strip()]

    for idx, item in enumerate(items):
        item_id = item.get("item_id", f"item_{idx}")
        prompt = item.get("prompt", "")
        ground_truth = item.get("ground_truth", "")
        plausible_wrong = item.get("plausible_wrong", "999")
        domain = item.get("domain", "math")

        print(f"--> Processing [{idx+1}/{len(items)}]: {item_id} (Domain: {domain})")
        print(f"    Prompt: {prompt[:85]}...")
        print(f"    Ground Truth: {ground_truth} | Adversary Target: {plausible_wrong}")

        try:
            res = await engine.run_matched_trajectory(
                item_id=item_id,
                prompt=prompt,
                ground_truth=ground_truth,
                domain=domain,
                replicate_id=0,
                R=2,
                byzantine_agent=attacker,
                enable_prompt_reminder=False,
                enable_verifier_gate=False,
                threat_config={
                    "knowledge": "oracle",
                    "onset": "post_anchor",
                    "adaptivity": "round_adaptive",
                    "objective": "targeted_error"
                },
                active_defense_name="none",
                oracle_wrong_target=plausible_wrong,
                temperature=0.7
            )
            trajectory_results.append(res)
            
            c_verdict = res["clean_consensus"]["verdict"]
            a_verdict = res["attacked_consensus"]["verdict"]
            k_adv = res["k_adv"]
            
            print(f"    [Seat Assignments] Adversarial Seat: Seat {k_adv}")
            print(f"    [Clean Branch Consensus]    Verdict: {c_verdict} (Majority: {res['clean_consensus']['is_majority']})")
            print(f"    [Attacked Branch Consensus] Verdict: {a_verdict} (Majority: {res['attacked_consensus']['is_majority']})")
            if str(c_verdict).strip() == str(ground_truth).strip() and str(a_verdict).strip() != str(ground_truth).strip():
                print(f"    >>> CAUSAL FLIP DETECTED: Attacker successfully swayed committee from {c_verdict} to {a_verdict}!\n")
            else:
                print(f"    >>> OUTCOME: Clean={c_verdict} | Attacked={a_verdict} | Ground Truth={ground_truth}\n")

        except Exception as e:
            logger.error(f"Execution failure on item {item_id}: {e}")

    # Telemetry and metrics reporting
    tracker = CostTracker()
    tracker.load_telemetry_file(raw_log_path)
    cost_summary = tracker.compute_summary()

    print("=======================================================")
    print(" LIVE DEMONSTRATION SUMMARY & TELEMETRY")
    print("=======================================================")
    print(f"Total API Calls Emitted   : {cost_summary['total_calls']}")
    print(f"Total Input Tokens        : {cost_summary['total_input_tokens']}")
    print(f"Total Output Tokens       : {cost_summary['total_output_tokens']}")
    print(f"Measured Total Spend      : ${cost_summary['total_cost_usd']:.4f} USD")
    print(f"Mean Latency Per Call     : {cost_summary['mean_latency_seconds']:.3f} s")
    print(f"Append-Only Telemetry Log : {raw_log_path}\n")

    if trajectory_results:
        t1 = analyze_tier1_operational(trajectory_results)
        t2 = analyze_tier2_mechanism(trajectory_results)
        print("--- OPERATIONAL & CAUSAL METRICS ---")
        print(f"Clean Committee Accuracy  : {t1.get('clean_accuracy'):.2%}")
        print(f"Attacked Committee Acc    : {t1.get('attacked_accuracy'):.2%}")
        print(f"Strict Majority Yield     : {t2.get('conditioned_population_size')}/{len(trajectory_results)}")
        print(f"Average Attack Effect AAE : {t2.get('AAE')}")
        print(f"Verdict Switch Rate (VSR) : {t2.get('VSR')}")
        print(f"Clean Collapse Rate (CCR) : {t2.get('CCR')}")
        
        if "contingency_matrix_df" in t2:
            print("\nMatched 2x2 Contingency Matrix:")
            print(t2["contingency_matrix_df"])

    print("\n=======================================================\n")

if __name__ == "__main__":
    asyncio.run(run_vertical_slice())
