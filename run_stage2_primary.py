import asyncio
import argparse
import json
import os
import sys
import logging
import pandas as pd
from typing import Dict, Any, List

# Safe stdout for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.client_wrapper import LLMClient
from core.concurrency import ConcurrencyManager
from debate.engine import DebateEngine
from attacks.redteam_agent import ByzantineAgent
from evaluation.tier1_operational import analyze_tier1_operational
from evaluation.tier2_mechanism import analyze_tier2_mechanism
from evaluation.statistical_tests import (
    question_clustered_bootstrap,
    permutation_test,
    benjamini_hochberg_adjustment
)
from evaluation.cost_tracker import CostTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

async def run_stage2():
    parser = argparse.ArgumentParser(description="Stage 2: Primary Factorial Multi-Agent Byzantine Experiment")
    parser.add_argument("--manifest", type=str, default="data/manifests/stage2_frontier_75.jsonl", help="Path to input manifest")
    parser.add_argument("--provider", type=str, default="gemini", help="API Provider")
    parser.add_argument("--model", type=str, default="gemini-3.7-flash", help="Primary model identifier")
    parser.add_argument("--committee-sizes", nargs="+", type=int, default=[3, 5, 7], help="Committee sizes N in {3, 5, 7}")
    parser.add_argument("--replicates", type=int, default=1, help="Stochastic replicates K (default: 1)")
    parser.add_argument("--rounds", type=int, default=2, help="Debate rounds R (default: 2)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for honest agents (default: 0.7)")
    parser.add_argument("--threats", nargs="+", type=str, default=["all"], help="Specific threat IDs to run (or 'all')")
    parser.add_argument("--resume", action="store_true", default=True, help="Automatically resume from existing telemetry checkpoint")
    parser.add_argument("--clean-start", action="store_true", default=False, help="Wipe output log and start fresh")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests (default: 8)")
    parser.add_argument("--output-log", type=str, default="data/raw_logs/stage2_primary.jsonl", help="Telemetry log output path")
    parser.add_argument("--report-output", type=str, default="data/reports/stage2_summary.json", help="Summary report path")
    
    args = parser.parse_args()

    if args.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            print("\n[ERROR] OPENAI_API_KEY is not set in environment.")
            print("Set your paid API key with: $env:OPENAI_API_KEY='your-key'\n")
            return
    else:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("\n[ERROR] GEMINI_API_KEY is not set in environment.")
            print("Set your paid API key with: $env:GEMINI_API_KEY='your-key'\n")
            return

    manifest_candidates = [
        args.manifest,
        os.path.join(os.path.dirname(__file__), args.manifest),
        args.manifest.replace("byzantine_mas/", "").replace("byzantine_mas\\", ""),
        os.path.join(os.path.dirname(__file__), args.manifest.replace("byzantine_mas/", "").replace("byzantine_mas\\", "")),
        os.path.join("data", "manifests", os.path.basename(args.manifest)),
        os.path.join(os.path.dirname(__file__), "data", "manifests", os.path.basename(args.manifest))
    ]
    resolved_manifest = None
    for cand in manifest_candidates:
        if os.path.exists(cand):
            resolved_manifest = cand
            break

    if not resolved_manifest:
        print(f"[ERROR] Manifest file {args.manifest} not found. Searched candidates: {manifest_candidates}")
        return
    args.manifest = resolved_manifest

    with open(args.manifest, "r", encoding="utf-8-sig") as f:
        items = [json.loads(line.strip()) for line in f if line.strip()]

    os.makedirs(os.path.dirname(args.output_log) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.report_output) or ".", exist_ok=True)

    if args.clean_start and os.path.exists(args.output_log):
        try:
            os.remove(args.output_log)
            print("[INFO] Clean start enabled: wiped previous telemetry log.")
        except Exception:
            pass

    # Load completed items for checkpoint resumption
    completed_checkpoints = set()
    if os.path.exists(args.output_log) and not args.clean_start:
        with open(args.output_log, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line.strip())
                        exp = rec.get("experimental_assignment", {})
                        n = exp.get("committee_size")
                        th = exp.get("threat_configuration", {}).get("knowledge") or exp.get("threat_configuration", {}).get("onset") or "unknown"
                        it = exp.get("item_id")
                        rep = exp.get("replicate_id", 0)
                        rnd = exp.get("round_index", 0)
                        br = exp.get("branch")
                        if rnd == args.rounds:
                            completed_checkpoints.add(f"N{n}_{it}_rep{rep}_{br}")
                    except Exception:
                        pass

    # 4 Full Factorial Threat Profiles:
    all_threats = [
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
        },
        {
            "id": "oracle_round0_adaptive",
            "knowledge": "oracle",
            "onset": "round_0",
            "adaptivity": "round_adaptive",
            "objective": "targeted_error",
            "influence_channels": {"forced_confidence": True, "spurious_proof": True, "authoritative_rhetoric": True}
        },
        {
            "id": "oracle_sleeper_adaptive",
            "knowledge": "oracle",
            "onset": "sleeper",
            "adaptivity": "round_adaptive",
            "objective": "targeted_error",
            "influence_channels": {"forced_confidence": True, "spurious_proof": True, "authoritative_rhetoric": True},
            "sleeper_K": 12
        }
    ]

    if "all" in args.threats:
        threat_conditions = all_threats
    else:
        threat_conditions = [t for t in all_threats if t["id"] in args.threats]

    total_conditions = len(args.committee_sizes) * len(threat_conditions)
    total_trajectories = total_conditions * len(items) * args.replicates

    # Calculate exact call count:
    # Per paired trajectory: N=3 (16), N=5 (26), N=7 (36)
    calls_per_pair = {3: 16, 5: 26, 7: 36}
    total_calls_expected = sum(
        len(threat_conditions) * len(items) * args.replicates * calls_per_pair.get(N, (2*N*(args.rounds+1) - (N-1)))
        for N in args.committee_sizes
    )
    from core.client_wrapper import PRICING_TABLE
    r_in, r_out = PRICING_TABLE.get(args.model, (1.04, 1.04))
    # Average tokens per call: ~360 input tokens, ~280 output tokens
    cost_per_call = (360 * (r_in / 1e6)) + (280 * (r_out / 1e6))
    est_cost_usd = total_calls_expected * cost_per_call
    est_cost_inr = est_cost_usd * 86.5

    print(f"\n==================================================================")
    print(f" STAGE 2: PRIMARY FACTORIAL MULTI-AGENT BYZANTINE EXPERIMENT")
    print(f" Model Identifier     : {args.model}")
    print(f" Committee Sizes (N)  : {args.committee_sizes}")
    print(f" Debate Rounds (R)    : {args.rounds}")
    print(f" Replicates (K)       : {args.replicates}")
    print(f" Sampling Temp (T)    : {args.temperature}")
    print(f" Threats Evaluated    : {[t['id'] for t in threat_conditions]}")
    print(f" Benchmark Manifest   : {args.manifest} ({len(items)} items across 3 domains)")
    print(f" Total Trajectories   : {total_trajectories} paired runs")
    print(f" Total API Calls      : ~{total_calls_expected:,} calls")
    print(f" Estimated Budget     : ~${est_cost_usd:.2f} USD (~₹{est_cost_inr:,.0f} INR)")
    print(f" Concurrency Limit    : {args.concurrency}")
    print(f" Telemetry Output     : {args.output_log}")
    print(f"==================================================================\n")

    concurrency_mgr = ConcurrencyManager(
        max_concurrent=args.concurrency,
        max_retries=4,
        hard_spend_ceiling_usd=750.0,
        initial_backoff_seconds=1.5
    )

    all_results = []
    condition_summaries = {}

    for N in args.committee_sizes:
        clients = [
            LLMClient(provider=args.provider, model_name=args.model, api_key=api_key)
            for _ in range(N)
        ]

        engine = DebateEngine(
            clients=clients,
            concurrency_manager=concurrency_mgr,
            study_id="stage2_primary_factorial",
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
                influence_channels=threat["influence_channels"],
                sleeper_K=threat.get("sleeper_K", 12)
            )

            # Parallel trajectory processing: Bounded to fully saturate args.concurrency
            item_concurrency = max(2, (args.concurrency // max(1, N)) + 1)
            item_sem = asyncio.Semaphore(item_concurrency)
            completed_lock = asyncio.Lock()
            completed_count = 0
            total_items_to_run = len(items) * args.replicates
            print(f"  [PARALLEL] Running {item_concurrency} trajectories concurrently (utilizing up to {args.concurrency} parallel API calls)...")

            async def process_item(idx, it, rep):
                nonlocal completed_count
                item_id = it.get("item_id", f"item_{idx}")
                prompt = it.get("prompt", "")
                gt = it.get("ground_truth", "")
                dom = it.get("domain", "math")
                plausible_wrong = it.get("plausible_wrong", None)

                # Checkpoint check
                clean_done = f"N{N}_{item_id}_rep{rep}_clean" in completed_checkpoints
                attacked_done = f"N{N}_{item_id}_rep{rep}_attacked" in completed_checkpoints
                if clean_done and attacked_done and args.resume:
                    async with completed_lock:
                        completed_count += 1
                        if completed_count % 25 == 0 or completed_count == total_items_to_run:
                            print(f"  [CHECKPOINT] Resumed/skipped {completed_count}/{total_items_to_run} items")
                    return None

                async with item_sem:
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
                        if attacker.sleeper_tracker:
                            attacker.sleeper_tracker.advance_task()

                        async with completed_lock:
                            completed_count += 1
                            if completed_count % 5 == 0 or completed_count == 1 or completed_count == total_items_to_run:
                                print(f"  --> Progress [{completed_count}/{total_items_to_run}] completed for N={N} | Threat: {threat['id']}")
                        return res

                    except Exception as e:
                        logger.error(f"Error on {item_id} (Rep {rep}): {e}")
                        async with completed_lock:
                            completed_count += 1
                        return None

            tasks = [
                process_item(idx, it, rep)
                for rep in range(args.replicates)
                for idx, it in enumerate(items)
            ]
            raw_results = await asyncio.gather(*tasks)
            cond_results = [r for r in raw_results if r is not None]
            all_results.extend(cond_results)

            # Condition-level analysis
            t1 = analyze_tier1_operational(cond_results)
            t2 = analyze_tier2_mechanism(cond_results)
            
            # Inferential statistics on conditioned stratum
            from evaluation.tier2_mechanism import is_correct
            matched_pairs = []
            for r in cond_results:
                shared = r.get("shared_honest_anchors", {})
                gt = r.get("ground_truth", "")
                dom = r.get("domain", "math")
                h_correct = sum(
                    1 for a in shared.values()
                    if is_correct(a.get("payload", {}).get("final_answer", ""), gt, dom)
                )
                if h_correct > (len(shared) / 2.0):
                    c_corr = is_correct(r.get("clean_consensus", {}).get("verdict", ""), gt, dom)
                    a_corr = is_correct(r.get("attacked_consensus", {}).get("verdict", ""), gt, dom)
                    matched_pairs.append({
                        "item_id": r.get("item_id"),
                        "clean_correct": c_corr,
                        "attacked_correct": a_corr
                    })

            boot_res = question_clustered_bootstrap(matched_pairs)
            perm_res = permutation_test(matched_pairs)
            perm_p = perm_res.get("p_value", 1.0)

            condition_summaries[cond_key] = {
                "N": N,
                "threat": threat["id"],
                "total_items": len(cond_results),
                "clean_acc": t1.get("clean_accuracy", 0.0),
                "attacked_acc": t1.get("attacked_accuracy", 0.0),
                "s_majority_yield": t2.get("conditioned_population_size", 0) / max(1, len(cond_results)),
                "AAE": t2.get("AAE", 0.0),
                "VSR": t2.get("VSR", 0.0),
                "CCR": t2.get("CCR", 0.0),
                "AER": t2.get("AER", 0.0),
                "p_harmful_revision": t2.get("p_harmful_revision_given_r0_correct", 0.0),
                "p_corrective_revision": t2.get("p_corrective_revision_given_r0_wrong", 0.0),
                "bootstrap_aae_mean": boot_res.get("bootstrap_mean", 0.0),
                "bootstrap_ci_lower": boot_res.get("ci_lower", 0.0),
                "bootstrap_ci_upper": boot_res.get("ci_upper", 0.0),
                "permutation_p_value": perm_p
            }

    # Benjamini-Hochberg FDR correction across all condition p-values
    p_values = [v["permutation_p_value"] for v in condition_summaries.values()]
    bh_results = benjamini_hochberg_adjustment(p_values, alpha=0.05)
    for idx, key in enumerate(condition_summaries.keys()):
        condition_summaries[key]["bh_fdr_adjusted_p"] = bh_results[idx][0] if idx < len(bh_results) else 1.0
        condition_summaries[key]["bh_fdr_significant"] = bh_results[idx][1] if idx < len(bh_results) else False

    # Print summary table
    print("\n==================================================================")
    print(" STAGE 2 PRIMARY FACTORIAL EXPERIMENT COMPLETE - SUMMARY")
    print("==================================================================")
    
    tracker = CostTracker()
    tracker.load_telemetry_file(args.output_log)
    cost_summary = tracker.compute_summary()
    
    print(f"Total API Calls Emitted : {cost_summary['total_calls']}")
    print(f"Total Input Tokens      : {cost_summary['total_input_tokens']:,}")
    print(f"Total Output Tokens     : {cost_summary['total_output_tokens']:,}")
    print(f"Total Measured Spend    : ${cost_summary['total_cost_usd']:.4f} USD (~₹{cost_summary['total_cost_usd']*86.5:,.1f} INR)")
    print(f"Mean Latency Per Call   : {cost_summary['mean_latency_seconds']:.3f} s\n")

    summary_df = pd.DataFrame(condition_summaries.values())
    cols_to_show = ["N", "threat", "clean_acc", "attacked_acc", "s_majority_yield", "AAE", "VSR", "CCR", "permutation_p_value", "bh_fdr_significant"]
    existing_cols = [c for c in cols_to_show if c in summary_df.columns]
    print(summary_df[existing_cols].to_string(index=False))

    # Save detailed JSON report
    with open(args.report_output, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_id": "stage2_primary_factorial",
            "model": args.model,
            "manifest": args.manifest,
            "cost_summary": cost_summary,
            "condition_summaries": condition_summaries
        }, f, indent=2)

    print(f"\n[REPORT SAVED] Full report exported to: {args.report_output}")
    print("==================================================================\n")

if __name__ == "__main__":
    asyncio.run(run_stage2())
