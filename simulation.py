"""
Retention campaign simulation for budget optimization.

Compares targeted (Expected Loss-ranked) vs random intervention strategies
and quantifies revenue saved, ROI, and targeting efficiency.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def compute_expected_loss(
    churn_proba: np.ndarray,
    clv: np.ndarray,
) -> np.ndarray:
    """
    Compute Expected Revenue Loss per customer.

    Expected_Loss = Predicted_Churn_Probability × CLV

    Business interpretation: This represents the expected revenue we would
    lose if the customer churns, weighted by how likely they are to churn.
    Prioritizing high Expected Loss customers maximizes impact of retention spend.

    Parameters
    ----------
    churn_proba : np.ndarray
        Predicted probability of churn (0-1).
    clv : np.ndarray
        Customer Lifetime Value for each customer.

    Returns
    -------
    np.ndarray
        Expected loss per customer.
    """
    return churn_proba * clv


def simulate_retention_campaign(
    expected_loss: np.ndarray,
    clv: np.ndarray,
    churn_proba: np.ndarray,
    budget: float,
    cost_per_intervention: float,
    churn_reduction_pct: float,
    strategy: str = "targeted",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Simulate a retention campaign under fixed budget constraints.

    Targeted strategy: Rank customers by Expected Loss (descending), intervene
    on top N customers until budget is exhausted.

    Random strategy: Randomly select N customers to intervene on.

    Assumption: Each intervention reduces the customer's churn probability
    by churn_reduction_pct (e.g., 0.15 = 15% relative reduction).

    Parameters
    ----------
    expected_loss : np.ndarray
        Expected revenue loss per customer.
    clv : np.ndarray
        Customer Lifetime Value.
    churn_proba : np.ndarray
        Predicted churn probability.
    budget : float
        Total marketing budget.
    cost_per_intervention : float
        Cost to reach one customer.
    churn_reduction_pct : float
        Assumed reduction in churn probability when targeted (e.g., 0.15).
    strategy : str
        'targeted' or 'random'.
    random_state : int
        For reproducible random selection.

    Returns
    -------
    dict with keys: n_targeted, revenue_saved, cost, roi, customers_targeted_indices
    """
    n_customers = len(expected_loss)
    max_targetable = int(budget / cost_per_intervention)
    n_targeted = min(max_targetable, n_customers)

    rng = np.random.default_rng(random_state)

    if strategy == "targeted":
        rank_order = np.argsort(-expected_loss)
        targeted_idx = rank_order[:n_targeted]
    else:
        targeted_idx = rng.choice(n_customers, size=n_targeted, replace=False)

    # Revenue saved: for each targeted customer, we reduce their churn prob by churn_reduction_pct
    # So new_churn_prob = churn_proba * (1 - churn_reduction_pct)
    # Revenue saved per customer = (old_churn_prob - new_churn_prob) * CLV
    # = churn_proba * churn_reduction_pct * CLV
    reduction_per_customer = churn_proba[targeted_idx] * churn_reduction_pct * clv[targeted_idx]
    revenue_saved = np.sum(reduction_per_customer)
    cost = n_targeted * cost_per_intervention
    roi = (revenue_saved - cost) / cost if cost > 0 else 0.0

    return {
        "n_targeted": n_targeted,
        "revenue_saved": revenue_saved,
        "cost": cost,
        "roi": roi,
        "targeted_indices": targeted_idx,
    }


def run_comparison(
    churn_proba: np.ndarray,
    clv: np.ndarray,
    budget: float,
    cost_per_intervention: float,
    churn_reduction_pct: float,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """
    Compare targeted vs random retention strategies.

    Parameters
    ----------
    churn_proba, clv : np.ndarray
    budget, cost_per_intervention, churn_reduction_pct : float
    random_state : int

    Returns
    -------
    targeted_results : dict
    random_results : dict
    efficiency_ratio : float
        Revenue saved (targeted) / Revenue saved (random). >1 means targeted is better.
    """
    expected_loss = compute_expected_loss(churn_proba, clv)

    targeted = simulate_retention_campaign(
        expected_loss, clv, churn_proba,
        budget, cost_per_intervention, churn_reduction_pct,
        strategy="targeted", random_state=random_state,
    )
    random = simulate_retention_campaign(
        expected_loss, clv, churn_proba,
        budget, cost_per_intervention, churn_reduction_pct,
        strategy="random", random_state=random_state,
    )

    efficiency = targeted["revenue_saved"] / random["revenue_saved"] if random["revenue_saved"] > 0 else float("inf")
    return targeted, random, efficiency


def budget_sweep(
    churn_proba: np.ndarray,
    clv: np.ndarray,
    budget_values: np.ndarray,
    cost_per_intervention: float,
    churn_reduction_pct: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simulate revenue saved across a range of budget levels.

    Used for the Revenue Saved vs Budget simulation plot.

    Parameters
    ----------
    churn_proba, clv : np.ndarray
    budget_values : np.ndarray
        Array of budget levels to simulate.
    cost_per_intervention, churn_reduction_pct : float
    random_state : int

    Returns
    -------
    pd.DataFrame with columns: budget, revenue_saved_targeted, revenue_saved_random, roi_targeted, roi_random
    """
    rows = []
    for budget in budget_values:
        targeted, random, _ = run_comparison(
            churn_proba, clv, budget, cost_per_intervention, churn_reduction_pct, random_state
        )
        rows.append({
            "budget": budget,
            "revenue_saved_targeted": targeted["revenue_saved"],
            "revenue_saved_random": random["revenue_saved"],
            "roi_targeted": targeted["roi"],
            "roi_random": random["roi"],
        })
    return pd.DataFrame(rows)


def sensitivity_analysis(
    churn_proba: np.ndarray,
    clv: np.ndarray,
    budget: float,
    churn_reduction_values: list,
    cost_per_intervention_values: list,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sensitivity analysis: vary churn_reduction_pct and cost_per_intervention.

    Returns revenue saved and ROI for targeted strategy across all parameter
    combinations. Helps assess robustness of recommendations.

    Parameters
    ----------
    churn_proba, clv : np.ndarray
    budget : float
    churn_reduction_values : list
        Values of churn_reduction_pct to sweep (e.g., [0.05, 0.10, 0.15, 0.20]).
    cost_per_intervention_values : list
        Values of cost_per_intervention to sweep (e.g., [25, 50, 75]).
    random_state : int

    Returns
    -------
    pd.DataFrame with columns: churn_reduction_pct, cost_per_intervention,
        revenue_saved_targeted, revenue_saved_random, roi_targeted, roi_random, efficiency
    """
    rows = []
    for churn_red in churn_reduction_values:
        for cost in cost_per_intervention_values:
            targeted, random, efficiency = run_comparison(
                churn_proba, clv, budget, cost, churn_red, random_state
            )
            rows.append({
                "churn_reduction_pct": churn_red,
                "cost_per_intervention": cost,
                "revenue_saved_targeted": targeted["revenue_saved"],
                "revenue_saved_random": random["revenue_saved"],
                "roi_targeted": targeted["roi"],
                "roi_random": random["roi"],
                "efficiency": efficiency,
            })
    return pd.DataFrame(rows)


def budget_scaling_analysis(
    churn_proba: np.ndarray,
    clv: np.ndarray,
    budget_values: np.ndarray,
    cost_per_intervention: float,
    churn_reduction_pct: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Budget scaling analysis: how revenue saved, ROI, and efficiency scale with budget.

    Extends budget_sweep with scaling metrics: n_targeted, efficiency, revenue_per_dollar.

    Parameters
    ----------
    churn_proba, clv : np.ndarray
    budget_values : np.ndarray
    cost_per_intervention, churn_reduction_pct : float
    random_state : int

    Returns
    -------
    pd.DataFrame with columns: budget, n_targeted, revenue_saved_targeted, revenue_saved_random,
        roi_targeted, roi_random, efficiency, revenue_per_dollar_targeted, revenue_per_dollar_random
    """
    rows = []
    for budget in budget_values:
        targeted, random, efficiency = run_comparison(
            churn_proba, clv, budget, cost_per_intervention, churn_reduction_pct, random_state
        )
        cost = targeted["cost"]
        rev_targeted = targeted["revenue_saved"]
        rev_random = random["revenue_saved"]

        revenue_per_dollar_targeted = rev_targeted / cost if cost > 0 else 0
        revenue_per_dollar_random = rev_random / cost if cost > 0 else 0

        rows.append({
            "budget": budget,
            "n_targeted": targeted["n_targeted"],
            "revenue_saved_targeted": rev_targeted,
            "revenue_saved_random": rev_random,
            "roi_targeted": targeted["roi"],
            "roi_random": random["roi"],
            "efficiency": efficiency,
            "revenue_per_dollar_targeted": revenue_per_dollar_targeted,
            "revenue_per_dollar_random": revenue_per_dollar_random,
        })

    return pd.DataFrame(rows)
