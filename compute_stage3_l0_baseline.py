import json
import os
import sys
import argparse
import pandas as pd
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from evaluation.tier1_operational import analyze_tier1_operational
from evaluation.tier2_mechanism import analyze_tier2_mechanism, is_correct
from evaluation.statistical_tests import question_clustered_bootstrap

def determine_majority(votes: List[str]) -> Dict[str, Any]:
    if not votes:
        return {"verdict": "no_verdict", "is_majority": False, "vote_count": 0}
    counts: Dict[str, int] = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    sorted_votes = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    winner, max_count = sorted_votes[0]
    is_majority = max_count > (len(votes) / 2.0)
    return {
        "verdict": winner if is_majority else "tie",
        "is_majority": is_majority,
        "vote_count": max_count
    }

def compute_l0_baseline(
    raw_log_path: str = "data/raw_logs/stage2_primary.jsonl",
    manifest_path: str = "data/manifests/stage2_llama_110.jsonl",
    target_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    target_committee_size: int = 3,
    output_report_path: str = "data/reports/stage3_l0_baseline.json"
):
    print("\n=======================================================")
    print(" STAGE 3: L0 (HARD LOCK) ZERO-COST ABLATION EXTRACTOR")
    print(f" Source Telemetry Log : {raw_log_path}")
    print(f" Target Model Filter  : {target_model}")
    print(f" Committee Size Filter: N={target_committee_size}")
    print("=======================================================\n")

    if not os.path.exists(raw_log_path):
        print(f"[ERROR] Telemetry file not found: {raw_log_path}")
        return

    # Load ground truths and domains from manifest
    gt_map = {}
    dom_map = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    iid = item.get("item_id")
                    gt_map[iid] = item.get("ground_truth", "")
                    dom_map[iid] = item.get("domain", "math")

    # Group Round 0 master anchors by (item_id, replicate_id)
    r0_clean_votes: Dict[tuple, Dict[int, str]] = {}

    print("--> Scanning telemetry log for frozen Round 0 master anchors...")
    with open(raw_log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line.strip())
                exp = rec.get("experimental_assignment", {})
                req = rec.get("model_request", {})

                if req.get("model_identifier") != target_model:
                    continue
                if exp.get("committee_size") != target_committee_size:
                    continue
                if exp.get("round_index") != 0 or exp.get("branch") != "clean":
                    continue

                iid = exp.get("item_id")
                rep = exp.get("replicate_id", 0)
                seat = exp.get("committee_seat", 0)
                ans = rec.get("agent_payload", {}).get("final_answer", "")

                key = (iid, rep)
                if key not in r0_clean_votes:
                    r0_clean_votes[key] = {}
                r0_clean_votes[key][seat] = ans

            except Exception:
                pass

    print(f"--> Found {len(r0_clean_votes)} matched Round 0 master items.")

    l0_results = []
    for (iid, rep), votes in r0_clean_votes.items():
        if len(votes) != target_committee_size or iid not in gt_map:
            continue

        gt = gt_map[iid]
        dom = dom_map.get(iid, "math")

        clean_votes = [votes[s] for s in range(target_committee_size)]
        cons = determine_majority(clean_votes)

        # Under L0 Hard Lock: Clean and Attacked branches are identical (nobody can change)
        l0_results.append({
            "item_id": iid,
            "replicate_id": rep,
            "ground_truth": gt,
            "domain": dom,
            "clean_consensus": cons,
            "attacked_consensus": cons,
            "shared_honest_anchors": {
                f"agent_{s}": {"payload": {"final_answer": votes[s]}}
                for s in range(1, target_committee_size)
            }
        })

    if not l0_results:
        print("[ERROR] No complete trajectories found.")
        return

    t1 = analyze_tier1_operational(l0_results)
    t2 = analyze_tier2_mechanism(l0_results)

    matched_pairs = []
    for r in l0_results:
        gt = r.get("ground_truth", "")
        dom = r.get("domain", "math")
        shared = r.get("shared_honest_anchors", {})
        h_corr = sum(
            1 for a in shared.values()
            if is_correct(a.get("payload", {}).get("final_answer", ""), gt, dom)
        )
        if h_corr > (len(shared) / 2.0):
            c_corr = is_correct(r.get("clean_consensus", {}).get("verdict", ""), gt, dom)
            a_corr = is_correct(r.get("attacked_consensus", {}).get("verdict", ""), gt, dom)
            matched_pairs.append({
                "item_id": r.get("item_id"),
                "clean_correct": c_corr,
                "attacked_correct": a_corr
            })

    boot_res = question_clustered_bootstrap(matched_pairs) if matched_pairs else {}

    summary = {
        "defense_id": "L0_hard_lock_ablation",
        "model": target_model,
        "committee_size": target_committee_size,
        "total_evaluated_items": len(l0_results),
        "total_cost_usd": 0.0,
        "tier1_operational": {
            "clean_accuracy": t1.get("clean_accuracy", 0.0),
            "attacked_accuracy": t1.get("attacked_accuracy", 0.0),
            "clean_terminal_error": t1.get("clean_terminal_error", 0.0),
            "attacked_terminal_error": t1.get("attacked_terminal_error", 0.0),
            "tie_rate": t1.get("clean_tie_rate", 0.0)
        },
        "tier2_causal_mechanism": {
            "conditioned_population_size_m": t2.get("conditioned_population_size", 0),
            "AAE": t2.get("AAE", 0.0),
            "VSR": t2.get("VSR", 0.0),
            "CCR": t2.get("CCR", 0.0),
            "AER": t2.get("AER", 0.0),
            "contingency_table": t2.get("contingency_table", {}),
            "bootstrap_ci_95": [boot_res.get("ci_lower", 0.0), boot_res.get("ci_upper", 0.0)]
        }
    }

    os.makedirs(os.path.dirname(output_report_path) or ".", exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("==================================================================")
    print(" L0 (HARD LOCK) ACCURATELY FILTERED TO LLAMA 3.3 70B ($0.00 SPENT)")
    print("==================================================================")
    print(f"Total Trajectories Analyzed : {len(l0_results)}")
    print(f"Eligible Stratum Pairs (m)  : {t2.get('conditioned_population_size', 0)}")
    print(f"Round 0 / L0 Panel Accuracy : {t1.get('clean_accuracy', 0.0):.2%} (Clean == Attacked)")
    print(f"Average Attack Effect (AAE) : {t2.get('AAE', 0.0):.4f}")
    print(f"Verdict Switch Rate (VSR)   : {t2.get('VSR', 0.0):.4f}")
    print(f"Clean Condition Error (CCR) : {t2.get('CCR', 0.0):.4f}")
    print(f"Attacked Error Rate (AER)   : {t2.get('AER', 0.0):.4f}")
    print(f"95% Bootstrap CI for AAE    : [{boot_res.get('ci_lower', 0.0):.4f}, {boot_res.get('ci_upper', 0.0):.4f}]")
    print(f"Saved Report                : {output_report_path}")
    print("==================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-log", type=str, default="data/raw_logs/stage2_primary.jsonl")
    parser.add_argument("--manifest", type=str, default="data/manifests/stage2_llama_110.jsonl")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    parser.add_argument("--committee-size", type=int, default=3)
    parser.add_argument("--output-report", type=str, default="data/reports/stage3_l0_baseline.json")
    args = parser.parse_args()

    compute_l0_baseline(
        raw_log_path=args.raw_log,
        manifest_path=args.manifest,
        target_model=args.model,
        target_committee_size=args.committee_size,
        output_report_path=args.output_report
    )
