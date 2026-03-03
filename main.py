"""
CLV Prediction and Retention Budget Optimization — Main Pipeline.

Runs the full analytics workflow: preprocessing, modeling, simulation, and visualization.
Execute this script to reproduce all results and generate outputs.
"""

from pathlib import Path

import numpy as np

import config
from preprocessing import preprocess_pipeline
from modeling import train_and_evaluate, save_model
from simulation import run_comparison, budget_sweep, sensitivity_analysis, budget_scaling_analysis
from visualization import (
    plot_roc_curves,
    plot_feature_importance,
    plot_revenue_vs_budget,
    plot_confusion_matrix,
    plot_sensitivity_analysis,
    plot_budget_scaling,
)


def main():
    """Execute the full CLV prediction and retention optimization pipeline."""
    print("=" * 60)
    print("CLV Prediction and Retention Budget Optimization")
    print("=" * 60)

    # --- 1. Preprocessing ---
    print("\n[1/6] Loading and preprocessing data...")
    df, X, y = preprocess_pipeline(config.DATA_PATH, handle_missing="drop")
    print(f"      Samples: {len(df):,} | Features: {X.shape[1]} | Churn rate: {y.mean():.2%}")

    # --- 2. Modeling ---
    print("\n[2/6] Training and evaluating models...")
    results, test_data = train_and_evaluate(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    # Select best model by ROC-AUC for simulation
    lr_auc = results["lr"]["roc_auc"]
    rf_auc = results["rf"]["roc_auc"]
    best_key = "rf" if rf_auc >= lr_auc else "lr"
    best_name = results[best_key]["model_name"]
    print(f"      Logistic Regression AUC: {lr_auc:.3f}")
    print(f"      Random Forest AUC: {rf_auc:.3f}")
    print(f"      Best model for simulation: {best_name}")

    # --- 3. Simulation ---
    print("\n[3/6] Running retention campaign simulation...")
    test_idx = test_data["X_test"].index
    df_test = df.loc[test_idx]
    clv_test = df_test["CLTV"].values
    churn_proba = results[best_key]["y_proba"]

    targeted, random, efficiency = run_comparison(
        churn_proba, clv_test,
        budget=config.DEFAULT_BUDGET,
        cost_per_intervention=config.DEFAULT_COST_PER_INTERVENTION,
        churn_reduction_pct=config.DEFAULT_CHURN_REDUCTION_PCT,
        random_state=config.RANDOM_STATE,
    )

    print(f"      Targeted: Revenue saved = ${targeted['revenue_saved']:,.0f}, ROI = {targeted['roi']:.2%}")
    print(f"      Random:   Revenue saved = ${random['revenue_saved']:,.0f}, ROI = {random['roi']:.2%}")
    print(f"      Targeting efficiency: {efficiency:.2f}x (targeted vs random)")

    # Budget sweep for visualization
    budget_values = np.linspace(5000, 100000, 20)
    sweep_df = budget_sweep(
        churn_proba, clv_test,
        budget_values=budget_values,
        cost_per_intervention=config.DEFAULT_COST_PER_INTERVENTION,
        churn_reduction_pct=config.DEFAULT_CHURN_REDUCTION_PCT,
        random_state=config.RANDOM_STATE,
    )

    # Sensitivity analysis: vary churn reduction and cost per intervention
    print("\n[3b] Running sensitivity analysis...")
    sensitivity_df = sensitivity_analysis(
        churn_proba, clv_test,
        budget=config.DEFAULT_BUDGET,
        churn_reduction_values=config.SENSITIVITY_CHURN_REDUCTION,
        cost_per_intervention_values=config.SENSITIVITY_COST_PER_INTERVENTION,
        random_state=config.RANDOM_STATE,
    )

    # Budget scaling analysis: how metrics scale with budget
    print("\n[3c] Running budget scaling analysis...")
    budget_scaling_values = np.linspace(
        config.BUDGET_SCALING_MIN,
        config.BUDGET_SCALING_MAX,
        config.BUDGET_SCALING_N_POINTS,
    )
    scaling_df = budget_scaling_analysis(
        churn_proba, clv_test,
        budget_values=budget_scaling_values,
        cost_per_intervention=config.DEFAULT_COST_PER_INTERVENTION,
        churn_reduction_pct=config.DEFAULT_CHURN_REDUCTION_PCT,
        random_state=config.RANDOM_STATE,
    )

    # --- 4. Save models ---
    print("\n[4/6] Saving models...")
    save_model(results["lr"]["model"], config.MODELS_DIR / "logistic_regression.pkl")
    save_model(results["rf"]["model"], config.MODELS_DIR / "random_forest.pkl")
    save_model(results["lr"]["scaler"], config.MODELS_DIR / "scaler.pkl")

    # --- 5. Visualizations ---
    print("\n[5/6] Generating figures...")
    plot_roc_curves(results, save_path=config.FIGURES_DIR / "roc_curves.png")
    plot_roc_curves(results, save_path=config.FIGURES_DIR / "roc_curves_poster.png", poster_mode=True)
    plot_feature_importance(
        results["rf"]["feature_importance"],
        top_n=15,
        save_path=config.FIGURES_DIR / "feature_importance.png",
        title="Feature Importance (Random Forest)",
    )
    plot_feature_importance(
        results["rf"]["feature_importance"],
        top_n=8,
        save_path=config.FIGURES_DIR / "feature_importance_poster.png",
        title="Feature Importance",
        poster_mode=True,
    )
    plot_revenue_vs_budget(sweep_df, save_path=config.FIGURES_DIR / "revenue_vs_budget.png")
    plot_revenue_vs_budget(sweep_df, save_path=config.FIGURES_DIR / "revenue_vs_budget_poster.png", poster_mode=True)
    plot_confusion_matrix(
        results[best_key]["confusion_matrix"],
        best_name,
        save_path=config.FIGURES_DIR / "confusion_matrix.png",
    )
    plot_sensitivity_analysis(
        sensitivity_df,
        metric="revenue_saved_targeted",
        save_path=config.FIGURES_DIR / "sensitivity_revenue.png",
    )
    plot_sensitivity_analysis(
        sensitivity_df,
        metric="roi_targeted",
        save_path=config.FIGURES_DIR / "sensitivity_roi.png",
    )
    plot_budget_scaling(
        scaling_df,
        save_path=config.FIGURES_DIR / "budget_scaling.png",
    )

    # --- 6. Summary report ---
    print("\n[6/6] Summary report")
    print("SUMMARY REPORT")
    print("=" * 60)
    print(f"\nModel Performance ({best_name}):")
    print(f"  ROC-AUC:   {results[best_key]['roc_auc']:.3f}")
    print(f"  Precision: {results[best_key]['precision']:.3f}")
    print(f"  Recall:    {results[best_key]['recall']:.3f}")
    print(f"\nRetention Campaign (Budget=${config.DEFAULT_BUDGET:,}, Cost=${config.DEFAULT_COST_PER_INTERVENTION}/customer):")
    print(f"  Revenue saved (targeted): ${targeted['revenue_saved']:,.0f}")
    print(f"  Revenue saved (random):   ${random['revenue_saved']:,.0f}")
    print(f"  ROI improvement: {(targeted['roi'] - random['roi']) / max(abs(random['roi']), 1e-6) * 100:.1f}%")
    print(f"  Targeting efficiency: {efficiency:.2f}x")
    print(f"\nSensitivity: {len(sensitivity_df)} parameter combinations | Budget scaling: {len(scaling_df)} budget levels")
    print(f"Outputs saved to: {config.OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
