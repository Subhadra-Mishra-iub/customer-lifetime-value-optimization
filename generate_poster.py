"""
Generate a 36" x 24" horizontal PDF poster for the CLV Prediction project.

Follows Disney Data & Analytics Women Award template.
Run after main.py. Output: outputs/poster.pdf
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

WIDTH, HEIGHT = 36, 24
# Tighter margins = wider boxes; symmetric layout
MARGIN_LR = 0.35
MARGIN_BOTTOM = 0.35
GAP = 0.28

AUTHOR_NAME = "Subhadra Mishra"

# Attractive color palette
COLORS = {
    "banner": "#1a5276",
    "header": "#2980b9",
    "accent": "#3498db",
    "bg_light": "#f0f8ff",
    "border": "#b8d4e8",
    "text_dark": "#2c3e50",
}


def main():
    fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="#fafbfc")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")

    # Content width - symmetric
    content_width = WIDTH - 2 * MARGIN_LR
    col_w = (content_width - 2 * GAP) / 3
    col1_x = MARGIN_LR
    col2_x = MARGIN_LR + col_w + GAP
    col3_x = MARGIN_LR + 2 * (col_w + GAP)

    # ---- TITLE BANNER (with header space above title) ----
    title_h = 2.0
    ax.add_patch(Rectangle((0, HEIGHT - title_h), WIDTH, title_h, facecolor=COLORS["banner"], edgecolor="none"))
    # Top padding: title starts 0.75" from top of banner
    ax.text(WIDTH/2, HEIGHT - 0.75, "Customer Lifetime Value (CLV) Prediction and\nRetention Budget Optimization Using Machine Learning",
            fontsize=26, fontweight="bold", ha="center", va="center", color="white")
    ax.text(WIDTH/2, HEIGHT - 1.55, AUTHOR_NAME, fontsize=18, ha="center", va="center", color="white")

    # ---- CONTENT ----
    content_top = HEIGHT - title_h - 0.45
    row_h = 3.7
    header_h = 0.5
    # Extra gap between content boxes and figures
    fig_gap = 0.75

    def add_section(x, y, w, h, title, lines, fontsize=13):
        """Section with header and body. Text inset to avoid overlap."""
        ax.add_patch(Rectangle((x, y + h - header_h), w, header_h, facecolor=COLORS["header"], edgecolor="none"))
        ax.text(x + w/2, y + h - header_h/2, title, fontsize=15, fontweight="bold", ha="center", va="center", color="white")
        ax.add_patch(FancyBboxPatch((x, y), w, h - header_h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=COLORS["bg_light"], edgecolor=COLORS["border"], linewidth=0.8))
        # Subtle left accent - narrow to avoid overlap
        ax.add_patch(Rectangle((x + 0.03, y + 0.04), 0.04, h - header_h - 0.08, facecolor=COLORS["accent"], alpha=0.25))
        body = "\n".join(lines)
        # Text inset: 0.4 from left to clear accent; stays within box
        ax.text(x + 0.4, y + h - header_h - 0.2, body, fontsize=fontsize, va="top", ha="left", color=COLORS["text_dark"])

    # LEFT COLUMN
    problem_lines = [
        "Telcos lose revenue from churn. Retention campaigns often waste budget on low-risk customers.",
        "",
        "• Predict churn probability",
        "• Estimate Expected Loss (P(churn) × CLV)",
        "• Optimize retention spend under budget",
        "• Compare targeted vs. random strategies",
    ]
    add_section(col1_x, content_top - row_h, col_w, row_h, "Background / Problem", problem_lines)

    methods_lines = [
        "1. Preprocessing: Load data, convert TotalCharges, map Churn to binary, one-hot encode.",
        "2. Modeling: Logistic Regression + Random Forest; ROC-AUC, precision, recall.",
        "3. Simulation: Expected Loss = P(churn)×CLV; targeted vs. random under fixed budget.",
        "4. Quantification: Revenue saved, ROI, targeting efficiency.",
    ]
    add_section(col1_x, content_top - 2*row_h, col_w, row_h, "Methods and Data", methods_lines, 13)

    # MIDDLE COLUMN
    results_lines = [
        "Model: ROC-AUC 0.851 | Precision 0.552 | Recall 0.735",
        "",
        "Simulation ($25K budget, $50/customer, 15% reduction):",
        "• Targeted: $224,684 saved | ROI 799%",
        "• Random: $116,083 saved | ROI 364%",
        "• Efficiency: 1.94×",
        "",
        "Sensitivity: robust across 5–25% churn reduction, $25–$100 cost.",
    ]
    add_section(col2_x, content_top - row_h, col_w, row_h, "Results", results_lines)

    refs_lines = [
        "IBM Telco Churn Dataset",
        "• github.com/IBM/telco-customer-churn-on-icp4d",
        "• Kaggle: waseemalastal/telco-customer-churn-ibm-dataset",
        "• License: Apache-2.0 | 7,043 customers, 31 features",
    ]
    add_section(col2_x, content_top - 2*row_h, col_w, row_h, "References", refs_lines, 12)

    # RIGHT COLUMN - Analysis/Conclusions with GitHub link
    conclusions_lines = [
        "• Rank by Expected Loss → maximize revenue saved under budget",
        "• Targeted strategy ~2× more revenue than random",
        "• Sensitivity + budget scaling: results robust",
        "• Reproducible pipeline: preprocessing, modeling, simulation",
        "",
        "Key takeaway: Budget scaling shows efficiency holds across $5K–$100K. Top drivers: tenure, monthly charges, contract.",
        "",
        "GitHub: github.com/Subhadra-Mishra-iub/customer-lifetime-value-optimization",
        "Reproducible Python pipeline for churn prediction and retention budget optimization.",
    ]
    add_section(col3_x, content_top - 2*row_h, col_w, 2*row_h, "Analysis / Conclusions", conclusions_lines, 12)

    # ---- FIGURES (with gap from content boxes) ----
    fig_zone_top = content_top - 2*row_h - fig_gap
    fig_h = fig_zone_top - MARGIN_BOTTOM - 0.35
    fig_w = (WIDTH - 2*MARGIN_LR - 2*GAP) / 3

    fig_files = [
        "roc_curves.png",
        "feature_importance_poster.png",
        "revenue_vs_budget.png",
    ]
    for i, fname in enumerate(fig_files):
        path = FIGURES_DIR / fname
        if not path.exists() and "poster" in fname:
            path = FIGURES_DIR / "feature_importance.png"
        x_pos = MARGIN_LR + i * (fig_w + GAP)
        if path.exists():
            img = plt.imread(path)
            ax_img = fig.add_axes([x_pos/WIDTH, MARGIN_BOTTOM/HEIGHT, fig_w/WIDTH, fig_h/HEIGHT])
            ax_img.imshow(img, aspect="auto")
            ax_img.axis("off")
        else:
            ax.add_patch(FancyBboxPatch((x_pos, MARGIN_BOTTOM), fig_w, fig_h, boxstyle="round,pad=0.02",
                                        facecolor=COLORS["bg_light"], edgecolor=COLORS["border"]))
            ax.text(x_pos + fig_w/2, MARGIN_BOTTOM + fig_h/2, f"Run main.py\nfor {fname}",
                    ha="center", va="center", fontsize=11)

    out_path = OUTPUT_DIR / "poster.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.12, facecolor="#fafbfc")
    plt.close()
    print(f"Poster saved to: {out_path}")


if __name__ == "__main__":
    main()
