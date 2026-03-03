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
# Symmetric layout with generous padding
MARGIN_LR = 0.4
MARGIN_BOTTOM = 0.4
GAP = 0.32
PAD_INNER = 0.7  # Internal padding inside section boxes for cleaner spacing

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

    # ---- TITLE BANNER ----
    title_h = 2.2
    ax.add_patch(Rectangle((0, HEIGHT - title_h), WIDTH, title_h, facecolor=COLORS["banner"], edgecolor="none"))
    ax.text(WIDTH/2, HEIGHT - 0.7, "Customer Lifetime Value Optimization Using Machine Learning",
            fontsize=24, fontweight="bold", ha="center", va="center", color="white")
    ax.text(WIDTH/2, HEIGHT - 1.15, AUTHOR_NAME, fontsize=16, ha="center", va="center", color="white")
    ax.text(WIDTH/2, HEIGHT - 1.55, "Data-driven retention targeting nearly doubles ROI under fixed marketing budgets.",
            fontsize=13, ha="center", va="center", color="white", style="italic")

    # ---- CONTENT ----
    content_top = HEIGHT - title_h - 0.45
    row_h = 3.7
    header_h = 0.5
    # Gap between content boxes and figures (smaller = taller charts)
    fig_gap = 0.5

    def add_section(x, y, w, h, title, lines, fontsize=12, line_spacing=0.38, blank_spacing=0.25, bottom_pad=0):
        """Section with header and body. Lines: str or (str, fontsize) for emphasis. bottom_pad adds breathing space at bottom."""
        ax.add_patch(Rectangle((x, y + h - header_h), w, header_h, facecolor=COLORS["header"], edgecolor="none"))
        ax.text(x + w/2, y + h - header_h/2, title, fontsize=14, fontweight="bold", ha="center", va="center", color="white")
        ax.add_patch(FancyBboxPatch((x, y), w, h - header_h, boxstyle="round,pad=0.06,rounding_size=0.18",
                                    facecolor=COLORS["bg_light"], edgecolor=COLORS["border"], linewidth=0.8))
        ax.add_patch(Rectangle((x + 0.04, y + 0.08), 0.03, h - header_h - 0.16, facecolor=COLORS["accent"], alpha=0.2))
        y0 = y + h - header_h - 0.4 + bottom_pad
        for item in lines:
            if isinstance(item, tuple):
                txt, fs = item[0], item[1]
                fw = item[2] if len(item) > 2 else "normal"
            else:
                txt, fs, fw = item, fontsize, "normal"
            if not txt:
                y0 -= blank_spacing
                continue
            ax.text(x + PAD_INNER, y0, txt, fontsize=fs, fontweight=fw, va="top", ha="left",
                    color=COLORS["text_dark"], wrap=True)
            y0 -= line_spacing

    # LEFT COLUMN - concise executive-style bullets
    problem_lines = [
        "• Churn drives revenue loss; campaigns miss high-value customers",
        "• Predict churn probability",
        "• Estimate Expected Loss (P(churn) × CLV)",
        "• Optimize spend under budget",
        "• Compare targeted vs. random strategies",
    ]
    add_section(col1_x, content_top - row_h, col_w, row_h, "Background / Problem", problem_lines)

    methods_lines = [
        "Data",
        "• IBM Telco: 7,043 customers, 31 features",
        "• Preprocessing: TotalCharges, Churn mapping, one-hot encode",
        "Modeling",
        "• Logistic Regression + Random Forest",
        "• Metrics: ROC-AUC, precision, recall",
        "Simulation",
        "• Expected Loss = P(churn) × CLV",
        "• Targeted vs. random under fixed budget",
        "• Revenue saved, ROI, targeting efficiency",
    ]
    add_section(col1_x, content_top - 2*row_h, col_w, row_h, "Methods", methods_lines, 11,
                line_spacing=0.30, blank_spacing=0.12, bottom_pad=0.25)

    # MIDDLE COLUMN - Results with bold key metrics (compact to prevent overspill)
    results_lines = [
        ("ROC-AUC: 0.851", 15, "bold"),
        ("Revenue Saved: $224,684", 15, "bold"),
        ("Efficiency: 1.94×", 15, "bold"),
        "",
        "Simulation ($25K budget, $50/customer, 15% reduction):",
        "Targeted: $224,684 saved | ROI 799%",
        "Random: $116,083 saved | ROI 364%",
        "",
        "• Sensitivity: robust across 5–25% churn, $25–$100 cost",
    ]
    add_section(col2_x, content_top - row_h, col_w, row_h, "Results", results_lines, 12,
                line_spacing=0.34, bottom_pad=0.2)

    refs_lines = [
        "• IBM Telco Churn Dataset",
        "• IBM: github.com/IBM/telco-customer-churn-on-icp4d",
        "• Kaggle: kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset",
        "• Apache-2.0 | 7,043 customers, 31 features",
    ]
    add_section(col2_x, content_top - 2*row_h, col_w, row_h, "References", refs_lines, 11)

    def _add_github_section(ax, x, y, w, h):
        """GitHub section: uniform layout, no wrap on links, Kaggle dataset link."""
        ax.add_patch(Rectangle((x, y + h - header_h), w, header_h, facecolor=COLORS["header"], edgecolor="none"))
        ax.text(x + w/2, y + h - header_h/2, "GitHub", fontsize=14, fontweight="bold", ha="center", va="center", color="white")
        ax.add_patch(FancyBboxPatch((x, y), w, h - header_h, boxstyle="round,pad=0.06,rounding_size=0.18",
                                    facecolor=COLORS["bg_light"], edgecolor=COLORS["border"], linewidth=0.8))
        ax.add_patch(Rectangle((x + 0.05, y + 0.1), 0.03, h - header_h - 0.2, facecolor=COLORS["accent"], alpha=0.2))
        repo_url = "https://github.com/Subhadra-Mishra-iub/customer-lifetime-value-optimization"
        kaggle_url = "https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset"
        # Repository - single line, no wrap
        ax.text(x + PAD_INNER, y + h - header_h - 0.35, "• Repository:", fontsize=11, va="top", ha="left",
                color=COLORS["text_dark"], fontweight="bold")
        t1 = ax.text(x + PAD_INNER, y + h - header_h - 0.62, "github.com/Subhadra-Mishra-iub/customer-lifetime-value-optimization",
                     fontsize=9, va="top", ha="left", color=COLORS["accent"], style="italic")
        if hasattr(t1, "set_url"):
            t1.set_url(repo_url)
        # Dataset - single line, no wrap
        ax.text(x + PAD_INNER, y + h - header_h - 0.92, "• Dataset:", fontsize=11, va="top", ha="left",
                color=COLORS["text_dark"], fontweight="bold")
        t2 = ax.text(x + PAD_INNER, y + h - header_h - 1.19, "kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset",
                     fontsize=9, va="top", ha="left", color=COLORS["accent"], style="italic")
        if hasattr(t2, "set_url"):
            t2.set_url(kaggle_url)
        # Description - on same line as heading, minimal gap
        ax.text(x + PAD_INNER, y + h - header_h - 1.46, "• Description:", fontsize=11, va="top", ha="left",
                color=COLORS["text_dark"], fontweight="bold")
        ax.text(x + PAD_INNER, y + h - header_h - 1.73, "Churn prediction and retention budget optimization pipeline.",
                fontsize=10, va="top", ha="left", color=COLORS["text_dark"], wrap=True)

    # RIGHT COLUMN - Analysis/Conclusions with Business Impact box
    conclusions_lines = [
        "• Rank by Expected Loss → maximize revenue saved under budget",
        "• Sensitivity + budget scaling: results robust",
        "",
        ("Business Impact", 14, "bold"),
        "• ~2× revenue improvement vs random",
        "• Scalable across budget levels",
        "• Reproducible ML + simulation pipeline",
        "",
        "Top drivers: tenure, monthly charges, contract.",
    ]
    add_section(col3_x, content_top - row_h, col_w, row_h, "Analysis / Conclusions", conclusions_lines, 12,
                line_spacing=0.30)

    # GitHub section: point-wise with Description heading, clickable link
    _add_github_section(ax, col3_x, content_top - 2*row_h, col_w, row_h)

    # ---- FIGURES (maximized for readability from 6 ft) ----
    fig_zone_top = content_top - 2*row_h - fig_gap
    fig_h = fig_zone_top - MARGIN_BOTTOM - 0.35
    fig_w = (WIDTH - 2*MARGIN_LR - 2*GAP) / 3

    fig_files = [
        "roc_curves_poster.png",
        "feature_importance_poster.png",
        "revenue_vs_budget_poster.png",
    ]
    for i, fname in enumerate(fig_files):
        path = FIGURES_DIR / fname
        if not path.exists():
            fallback = "roc_curves.png" if "roc" in fname else "revenue_vs_budget.png" if "revenue" in fname else "feature_importance.png"
            path = FIGURES_DIR / fallback
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
