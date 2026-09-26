import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import scipy.stats as stats

def question_clustered_bootstrap(
    item_effects: List[Dict[str, Any]],
    B: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Question-Clustered Non-Parametric Bootstrap carrying all nested replicates.
    Calculates empirical point estimates with Bias-Corrected and Accelerated (BCa) / Percentile 95% CIs.
    """
    if not item_effects:
        return {"observed_aae": 0.0, "bootstrap_mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0}

    rng = np.random.RandomState(seed)
    n = len(item_effects)

    deltas = []
    for it in item_effects:
        c = it.get("clean_correct", False)
        a = it.get("attacked_correct", False)
        if c and not a:
            deltas.append(1.0)
        elif not c and a:
            deltas.append(-1.0)
        else:
            deltas.append(0.0)

    deltas = np.array(deltas)
    obs_aae = float(np.mean(deltas))

    boot_means = []
    for _ in range(B):
        sample_indices = rng.choice(n, size=n, replace=True)
        boot_means.append(np.mean(deltas[sample_indices]))

    boot_means = np.array(boot_means)
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    p_val = float(np.mean(boot_means <= 0.0))

    return {
        "observed_aae": round(obs_aae, 4),
        "bootstrap_mean": round(float(np.mean(boot_means)), 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_val, 4),
        "B_resamples": B
    }

def permutation_test(
    item_effects: List[Dict[str, Any]],
    M: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Question-Cluster Level Paired Sign-Flip Permutation Test.
    """
    if not item_effects:
        return {"observed_statistic": 0.0, "p_value": 1.0}

    rng = np.random.RandomState(seed)
    n = len(item_effects)

    deltas = []
    for it in item_effects:
        c = it.get("clean_correct", False)
        a = it.get("attacked_correct", False)
        deltas.append(1.0 if (c and not a) else (-1.0 if (not c and a) else 0.0))

    deltas = np.array(deltas)
    obs_stat = np.abs(np.mean(deltas))

    perm_count = 0
    for _ in range(M):
        signs = rng.choice([-1.0, 1.0], size=n)
        if np.abs(np.mean(deltas * signs)) >= obs_stat:
            perm_count += 1

    return {
        "observed_statistic": round(float(obs_stat), 4),
        "p_value": round(float(perm_count / M), 5),
        "M_permutations": M
    }

def benjamini_hochberg_adjustment(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """
    Benjamini-Hochberg False Discovery Rate (BH-FDR) Multiplicity Adjustment.
    """
    m = len(p_values)
    if m == 0:
        return []

    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted = np.zeros(m)
    significant = np.zeros(m, dtype=bool)

    cum_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        adj_p = (sorted_p[i] * m) / rank
        cum_min = min(cum_min, adj_p)
        adjusted[i] = min(1.0, cum_min)

    significant = adjusted <= alpha

    # Restore original order
    orig_adjusted = np.zeros(m)
    orig_sig = np.zeros(m, dtype=bool)
    for i, orig_idx in enumerate(sorted_indices):
        orig_adjusted[orig_idx] = adjusted[i]
        orig_sig[orig_idx] = significant[i]

    return [(float(orig_adjusted[i]), bool(orig_sig[i])) for i in range(m)]

def marginal_gee(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Marginal GEE with logit link and exchangeable correlation clustered on Question ID:
    logit(P(E_ikb = 1)) = beta_0 + beta_1 Attack + beta_2 Timing + beta_3 Defence + beta_4 N + beta_5 Dataset + ...
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from statsmodels.genmod.cov_struct import Exchangeable

        formula = "is_error ~ C(branch, Treatment('clean'))"
        if "committee_size" in df.columns:
            formula += " + C(committee_size)"
        if "active_defence" in df.columns:
            formula += " + C(active_defence)"

        model = smf.gee(
            formula,
            groups="item_id",
            data=df,
            family=sm.families.Binomial(),
            cov_struct=Exchangeable()
        )
        fit_res = model.fit()
        return {
            "summary": str(fit_res.summary()),
            "params": fit_res.params.to_dict(),
            "pvalues": fit_res.pvalues.to_dict(),
            "converged": fit_res.converged
        }
    except Exception as e:
        return {"error": str(e), "converged": False}

def longitudinal_transitions_gee(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Binary Longitudinal Dynamics GEE with AR(1) working correlation and graceful fallback.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.cov_struct import Autoregressive, Exchangeable

    try:
        model = smf.gee(
            "is_correct ~ round_index * C(branch)",
            groups="item_id",
            time="round_index",
            data=df,
            family=sm.families.Binomial(),
            cov_struct=Autoregressive()
        )
        res = model.fit()
        if res.converged:
            return {"model": "GEE_AR1", "params": res.params.to_dict(), "pvalues": res.pvalues.to_dict(), "converged": True}
    except Exception:
        pass

    try:
        model_exch = smf.gee(
            "is_correct ~ round_index * C(branch)",
            groups="item_id",
            data=df,
            family=sm.families.Binomial(),
            cov_struct=Exchangeable()
        )
        res_exch = model_exch.fit()
        if res_exch.converged:
            return {"model": "GEE_Exchangeable_Fallback", "params": res_exch.params.to_dict(), "pvalues": res_exch.pvalues.to_dict(), "converged": True}
    except Exception:
        pass

    return {"model": "Failed", "converged": False}

def sleeper_trust_decay_model(trust_df: pd.DataFrame, K: int = 5) -> Dict[str, Any]:
    """
    Longitudinal Trust Decay & Sleeper Dynamics (Stage 4A):
    Piecewise LMM with knot at activation task K:
    Trust_{j,t} = beta_0 + beta_1 t + beta_2 I(t > K) + beta_3 (t - K)_+ + u_j + epsilon_{j,t}
    """
    try:
        import statsmodels.formula.api as smf

        df = trust_df.copy()
        df["is_activated_step"] = (df["task_index"] > K).astype(float)
        df["t_hinge"] = df["task_index"].apply(lambda t: max(0.0, float(t - K)))

        group_col = "sequence_id" if "sequence_id" in df.columns else "agent_id"
        lmm = smf.mixedlm(
            "trust_score ~ task_index + is_activated_step + t_hinge",
            data=df,
            groups=df[group_col]
        )
        res = lmm.fit()
        return {
            "beta_0_intercept": float(res.params.get("Intercept", 0.0)),
            "beta_1_pre_attack_accumulation": float(res.params.get("task_index", 0.0)),
            "beta_2_immediate_activation_penalty": float(res.params.get("is_activated_step", 0.0)),
            "beta_3_post_activation_decay": float(res.params.get("t_hinge", 0.0)),
            "pvalues": res.pvalues.to_dict(),
            "converged": res.converged
        }
    except Exception as e:
        return {"error": str(e), "converged": False}
