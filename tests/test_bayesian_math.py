import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from defenses.bayesian_filter import BayesianTrustFilter
from attacks.sleeper import SleeperAgentTracker
from evaluation.statistical_tests import (
    question_clustered_bootstrap,
    permutation_test,
    sleeper_trust_decay_model,
    benjamini_hochberg_adjustment
)

def test_bayesian_filter_equations():
    bf = BayesianTrustFilter(a0=1.0, b0=1.0, discount=0.85, threshold=0.35, min_obs=3)
    agent = "agent_0"
    assert bf.get_trust(agent) == 0.5
    assert bf.is_quarantined(agent) is False

    # Task 1: Honest / Correct (z=1.0)
    bf.update_trust(agent, 1.0)
    assert abs(bf.get_trust(agent) - (2.0 / 3.0)) < 1e-5
    assert bf.is_quarantined(agent) is False

    # Task 2: Honest / Correct (z=1.0)
    bf.update_trust(agent, 1.0)
    assert abs(bf.get_trust(agent) - (2.85 / 3.85)) < 1e-4

    # Task 3: Byzantine Attack / Wrong (z=0.0)
    bf.update_trust(agent, 0.0)
    assert bf.is_quarantined(agent) is False

    # Consecutive failures
    for _ in range(5):
        bf.update_trust(agent, 0.0)

    assert bf.get_trust(agent) < 0.35
    assert bf.is_quarantined(agent) is True
    assert agent in bf.get_quarantined_peers()

def test_sleeper_agent_tracker():
    tracker = SleeperAgentTracker(K=3)
    assert tracker.is_activated() is False
    assert tracker.advance_task() is False
    assert tracker.advance_task() is False
    assert tracker.advance_task() is False
    assert tracker.advance_task() is True
    assert tracker.is_activated() is True

def test_statistical_inference_functions():
    effects = []
    for i in range(100):
        c = (i % 10 != 0)
        a = (i % 3 == 0)
        effects.append({"item_id": f"item_{i}", "clean_correct": c, "attacked_correct": a})

    boot_res = question_clustered_bootstrap(effects, B=500, seed=42)
    assert "observed_aae" in boot_res
    assert "ci_lower" in boot_res
    assert "ci_upper" in boot_res
    assert boot_res["ci_lower"] <= boot_res["ci_upper"]

    perm_res = permutation_test(effects, M=1000, seed=42)
    assert "p_value" in perm_res
    assert 0.0 <= perm_res["p_value"] <= 1.0

    bh_res = benjamini_hochberg_adjustment([0.001, 0.04, 0.20], alpha=0.05)
    assert len(bh_res) == 3
    assert bh_res[0][1] is True

def test_sleeper_lmm_decay():
    rows = []
    for agent_id in ["agent_0", "agent_1", "agent_2"]:
        for t in range(1, 11):
            base = 0.5 + 0.05 * t if t <= 5 else 0.75 - 0.1 * (t - 5)
            noise = np.random.normal(0, 0.02)
            rows.append({
                "agent_id": agent_id,
                "task_index": t,
                "trust_score": max(0.0, min(1.0, base + noise))
            })

    df = pd.DataFrame(rows)
    lmm_res = sleeper_trust_decay_model(df, K=5)
    assert "converged" in lmm_res
    if lmm_res["converged"]:
        assert "beta_0_intercept" in lmm_res
        assert "beta_1_pre_attack_accumulation" in lmm_res
        assert "beta_2_immediate_activation_penalty" in lmm_res
        assert "beta_3_post_activation_decay" in lmm_res

if __name__ == "__main__":
    test_bayesian_filter_equations()
    test_sleeper_agent_tracker()
    test_statistical_inference_functions()
    test_sleeper_lmm_decay()
    print("All Bayesian and Statistical tests passed successfully!")
