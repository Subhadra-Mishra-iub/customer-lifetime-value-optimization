# Customer Lifetime Value (CLV) Prediction and Retention Budget Optimization

Predicts customer churn, estimates CLV, and simulates retention campaigns to optimize marketing spend.

## Business Objective

- **Predict** customer churn probability using Logistic Regression and Random Forest
- **Estimate** Customer Lifetime Value (CLV) from the dataset
- **Compute** Expected Revenue Loss per customer: `Expected_Loss = Churn_Probability × CLV`
- **Simulate** retention campaigns under fixed budget constraints
- **Compare** targeted (Expected Loss–ranked) vs random intervention strategies
- **Quantify** revenue saved, ROI improvement, and targeting efficiency

## Data

This project uses the **Telco customer churn: IBM dataset** (IBM Cognos Analytics base samples):

- **Source:** [Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)
- **Context:** A fictional telco company that provided home phone and Internet services to 7,043 customers in California in Q3
- **License:** See dataset page for license terms

Place `Telco_customer_churn.xlsx` in the project root (or update `DATA_PATH` in `config.py`).

## Project Structure

```
├── config.py           # Centralized paths and parameters
├── preprocessing.py    # Data loading, cleaning, feature engineering
├── modeling.py         # Model training and evaluation
├── simulation.py       # Retention campaign simulation
├── visualization.py   # ROC curves, feature importance, revenue plots
├── main.py             # Full pipeline orchestration
├── requirements.txt   # Python dependencies
├── outputs/            # Generated artifacts
│   ├── figures/        # ROC curve, feature importance, revenue vs budget
│   └── models/         # Saved models (pickle)
└── Telco_customer_churn.xlsx  # Dataset
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the full pipeline:

```bash
python main.py
```

Generate a 36"×24" PDF poster (run after `main.py`):

```bash
python generate_poster.py
```

Output: `outputs/poster.pdf`

**Before submission:** Edit `AUTHOR_NAME` in `generate_poster.py` with your name, then regenerate. See `SUBMISSION_CHECKLIST.md` for the full checklist.

This will:

1. Load and preprocess the Telco churn dataset
2. Train Logistic Regression and Random Forest models
3. Evaluate using ROC-AUC, confusion matrix, precision, and recall
4. Run retention campaign simulation (targeted vs random)
5. Generate figures in `outputs/figures/`
6. Save models in `outputs/models/`

## Configuration

Edit `config.py` to adjust:

- **Data path**: `DATA_PATH` (default: `Telco_customer_churn.xlsx` in project root)
- **Modeling**: `RANDOM_STATE`, `TEST_SIZE`
- **Simulation**: `DEFAULT_BUDGET`, `DEFAULT_COST_PER_INTERVENTION`, `DEFAULT_CHURN_REDUCTION_PCT`

## Outputs

| File | Description |
|------|-------------|
| `roc_curves.png` | ROC curves for both models |
| `feature_importance.png` | Top features driving churn (Random Forest) |
| `revenue_vs_budget.png` | Revenue saved vs budget: targeted vs random |
| `confusion_matrix.png` | Confusion matrix for best model |
| `sensitivity_revenue.png` | Sensitivity: revenue saved vs churn reduction % and cost |
| `sensitivity_roi.png` | Sensitivity: ROI vs churn reduction % and cost |
| `budget_scaling.png` | Budget scaling: revenue, ROI, and efficiency vs budget |
| `logistic_regression.pkl` | Saved Logistic Regression model |
| `random_forest.pkl` | Saved Random Forest model |

## Reproducibility

- Fixed random seeds (`RANDOM_STATE = 42`) throughout
- Paths in `config.py`; no hardcoded paths
- Modular structure
