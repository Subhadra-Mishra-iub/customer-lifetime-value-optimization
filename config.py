"""
Configuration for CLV Prediction and Retention Budget Optimization.

Centralizes paths and parameters to ensure reproducibility and avoid hardcoded values.
All paths are resolved relative to the project root.
"""

from pathlib import Path

# Project root (directory containing this config file)
PROJECT_ROOT = Path(__file__).resolve().parent

# Data paths
DATA_DIR = PROJECT_ROOT
DATA_FILE = "Telco_customer_churn.xlsx"
DATA_PATH = DATA_DIR / DATA_FILE

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

# Create output directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Modeling parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1  # For optional validation split

# Simulation parameters (can be overridden at runtime)
DEFAULT_BUDGET = 25000  # Fixed marketing budget in currency units
DEFAULT_COST_PER_INTERVENTION = 50  # Cost to reach one customer
DEFAULT_CHURN_REDUCTION_PCT = 0.15  # Assumed 15% reduction in churn probability when targeted

# Sensitivity analysis: parameter ranges to sweep
SENSITIVITY_CHURN_REDUCTION = [0.05, 0.10, 0.15, 0.20, 0.25]  # 5% to 25%
SENSITIVITY_COST_PER_INTERVENTION = [25, 50, 75, 100]  # Cost per customer

# Budget scaling: range and granularity
BUDGET_SCALING_MIN = 5000
BUDGET_SCALING_MAX = 100000
BUDGET_SCALING_N_POINTS = 25
