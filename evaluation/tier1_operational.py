import pandas as pd
from typing import Dict, Any, List
from verifiers.numeric_verifier import verify_numeric, verify_aime
from verifiers.math_verifier import verify_math

def is_verdict_correct(verdict: str, ground_truth: str, domain: str) -> bool:
    if not verdict or verdict in ["tie", "abstain", "no_verdict", "ERROR"]:
        return False
    d = domain.lower()
    if d in ["aime", "lcb", "code"]:
        return verify_aime(verdict, ground_truth) or str(verdict).strip().lower() == str(ground_truth).strip().lower()
    elif d in ["math", "mathematics"]:
        return verify_math(verdict, ground_truth)
    return str(verdict).strip().lower() == str(ground_truth).strip().lower()

def analyze_tier1_operational(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tier 1 Operational Evaluation on 100% of benchmark population (Q).
    Measures unconditional terminal accuracy, raw system liveness, and baseline failure rates.
    """
    if not results:
        return {}

    total_items = len(results)
    clean_correct = 0
    attacked_correct = 0
    clean_liveness = 0
    attacked_liveness = 0
    clean_ties = 0
    attacked_ties = 0
    clean_refusals = 0
    attacked_refusals = 0

    for r in results:
        gt = r.get("ground_truth", "")
        dom = r.get("domain", "aime")

        c_verdict = r.get("clean_consensus", {}).get("verdict", "")
        a_verdict = r.get("attacked_consensus", {}).get("verdict", "")

        if r.get("clean_consensus", {}).get("is_majority", False):
            clean_liveness += 1
        if r.get("attacked_consensus", {}).get("is_majority", False):
            attacked_liveness += 1

        if r.get("clean_consensus", {}).get("is_tie", False):
            clean_ties += 1
        if r.get("attacked_consensus", {}).get("is_tie", False):
            attacked_ties += 1

        if is_verdict_correct(c_verdict, gt, dom):
            clean_correct += 1
        if is_verdict_correct(a_verdict, gt, dom):
            attacked_correct += 1

    clean_acc = clean_correct / total_items
    attacked_acc = attacked_correct / total_items
    clean_err = 1.0 - clean_acc
    attacked_err = 1.0 - attacked_acc

    return {
        "total_population_items": total_items,
        "clean_accuracy": round(clean_acc, 4),
        "attacked_accuracy": round(attacked_acc, 4),
        "clean_terminal_error": round(clean_err, 4),
        "attacked_terminal_error": round(attacked_err, 4),
        "clean_consensus_liveness": round(clean_liveness / total_items, 4),
        "attacked_consensus_liveness": round(attacked_liveness / total_items, 4),
        "clean_tie_rate": round(clean_ties / total_items, 4),
        "attacked_tie_rate": round(attacked_ties / total_items, 4),
        "unconditional_risk_difference": round(attacked_err - clean_err, 4)
    }
