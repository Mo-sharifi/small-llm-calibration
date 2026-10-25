"""Analysis script: computes all metrics, outputs CSV/JSON tables, and generates figures."""

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import compute_all_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "results" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    ("qwen2.5_0.5b_instruct", "0.5B", 0.5),
    ("qwen2.5_1.5b_instruct", "1.5B", 1.5),
    ("qwen2.5_3b_instruct", "3.0B", 3.0),
    ("qwen2.5_7b_instruct", "7.0B", 7.0),
]

METHOD_KEYS = [
    ("conf_token_prob", "Token Prob"),
    ("conf_verbalized", "Verbalized"),
    ("conf_p_true", "P(True)"),
    ("conf_consistency", "Consistency"),
]


def load_raw_data() -> Dict[str, List[Dict[str, Any]]]:
    """Load JSONL files for each model."""
    data = {}
    for slug, label, size in MODEL_ORDER:
        file_path = RAW_DIR / f"{slug}.jsonl"
        if not file_path.exists():
            print(f"Warning: Missing {file_path}")
            continue
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        data[slug] = records
        print(f"Loaded {len(records)} records for {slug} ({label})")
    return data


def run_full_analysis():
    raw_data = load_raw_data()

    table_rows = []
    json_summary = {}

    datasets_to_eval = [("overall", "Overall (Combined)"), ("trivia_qa", "TriviaQA"), ("sciq", "SciQ")]

    for slug, label, size in MODEL_ORDER:
        records = raw_data.get(slug, [])
        if not records:
            continue

        json_summary[slug] = {"label": label, "size_b": size, "datasets": {}}

        for d_key, d_name in datasets_to_eval:
            if d_key == "overall":
                sub_records = records
            else:
                sub_records = [r for r in records if r["dataset"] == d_key]

            if not sub_records:
                continue

            labels = [bool(r["is_correct"]) for r in sub_records]
            acc = float(np.mean(labels))
            n_items = len(sub_records)

            dataset_dict = {"n": n_items, "accuracy": acc, "methods": {}}

            for conf_col, meth_label in METHOD_KEYS:
                confs = [float(r[conf_col]) for r in sub_records]
                metrics = compute_all_metrics(confs, labels, n_bins=10)

                table_rows.append({
                    "model_slug": slug,
                    "model_label": label,
                    "size_b": size,
                    "dataset": d_key,
                    "dataset_name": d_name,
                    "n_samples": n_items,
                    "accuracy": acc,
                    "method_key": conf_col,
                    "method": meth_label,
                    "auroc": metrics["auroc"],
                    "ece": metrics["ece"],
                    "brier": metrics["brier"],
                })

                dataset_dict["methods"][meth_label] = {
                    "auroc": metrics["auroc"],
                    "ece": metrics["ece"],
                    "brier": metrics["brier"],
                    "bin_details": metrics["bin_details"],
                }

            json_summary[slug]["datasets"][d_key] = dataset_dict

    # Save metrics table CSV and JSON
    df = pd.DataFrame(table_rows)
    csv_path = RESULTS_DIR / "metrics_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics CSV to {csv_path}")

    json_path = RESULTS_DIR / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)
    print(f"Saved metrics JSON to {json_path}")

    # Generate Figures
    generate_figures(df, json_summary, raw_data)

    # Print summary table to console
    print("\n" + "=" * 78)
    print("OVERALL CALIBRATION RESULTS (TriviaQA + SciQ)")
    print("=" * 78)
    overall_df = df[df["dataset"] == "overall"][
        ["model_label", "accuracy", "method", "auroc", "ece", "brier"]
    ]
    print(overall_df.to_string(index=False))
    print("=" * 78)


def generate_figures(df: pd.DataFrame, json_summary: Dict[str, Any], raw_data: Dict[str, Any]):
    """Generate publication-quality figures in PNG and PDF formats."""
    # Publication style: designed at exact physical paper dimensions (7.0 in wide for 2-column, 3.33 in for 1-column)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "figure.titlesize": 10.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 6.0,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
    })

    method_colors = {
        "Token Prob": "#1f77b4",     # Blue
        "Verbalized": "#d62728",     # Red
        "P(True)": "#2ca02c",        # Green
        "Consistency": "#ff7f0e",    # Orange
    }
    method_markers = {
        "Token Prob": "o",
        "Verbalized": "s",
        "P(True)": "^",
        "Consistency": "D",
    }

    x_pos = np.arange(len(MODEL_ORDER))
    x_labels = [label for _, label, _ in MODEL_ORDER]

    # -------------------------------------------------------------
    # Figure 1: AUROC vs Model Size (Full 2-column width: 7.0 in)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    datasets = [
        ("overall", "Combined (1,000 items)"),
        ("trivia_qa", "TriviaQA (500 items)"),
        ("sciq", "SciQ (500 items)"),
    ]

    for ax, (d_key, title) in zip(axes, datasets):
        sub_df = df[df["dataset"] == d_key]
        for meth_col, meth_label in METHOD_KEYS:
            meth_sub = sub_df[sub_df["method"] == meth_label]
            y_vals = []
            for slug, _, _ in MODEL_ORDER:
                row = meth_sub[meth_sub["model_slug"] == slug]
                y_vals.append(row["auroc"].values[0] if not row.empty else np.nan)

            ax.plot(
                x_pos,
                y_vals,
                label=meth_label,
                color=method_colors[meth_label],
                marker=method_markers[meth_label],
                linewidth=1.8,
                markersize=5.5,
                zorder=3,
            )
        ax.axhline(0.50, color="#666666", linestyle="--", linewidth=1.1, alpha=0.8, label="Chance (0.50)" if ax == axes[0] else "", zorder=2)
        ax.set_title(title, fontweight="bold", fontsize=9.5, pad=3)
        ax.set_xlabel("Model Scale", fontsize=8.5, labelpad=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontweight="bold", fontsize=8.0)
        ax.set_xlim(-0.25, len(x_pos) - 0.75)
        ax.set_ylim(0.35, 1.02)
        ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid(True, linestyle=":", alpha=0.5, zorder=1)
        if ax == axes[0]:
            ax.set_ylabel("AUROC (Discrimination)", fontsize=8.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cccccc",
        fontsize=7.5,
        handletextpad=0.3,
        columnspacing=1.0,
    )
    plt.subplots_adjust(top=0.83, bottom=0.17, left=0.08, right=0.98, wspace=0.12)
    fig1_png = FIGURES_DIR / "fig1_auroc_vs_size.png"
    fig1_pdf = FIGURES_DIR / "fig1_auroc_vs_size.pdf"
    plt.savefig(fig1_png, dpi=300)
    plt.savefig(fig1_pdf)
    plt.close()
    print(f"Generated {fig1_png} and {fig1_pdf}")

    # -------------------------------------------------------------
    # Figure 2: Reliability Diagrams (Full 2-column width: 7.0 in)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 2.7), sharey=True, sharex=True)
    target_models = [("qwen2.5_0.5b_instruct", "0.5B"), ("qwen2.5_7b_instruct", "7.0B")]

    for row_idx, (m_slug, m_name) in enumerate(target_models):
        m_data = json_summary[m_slug]["datasets"]["overall"]["methods"]
        for col_idx, (m_key, m_label) in enumerate(METHOD_KEYS):
            ax = axes[row_idx, col_idx]
            bins = m_data[m_label]["bin_details"]
            ece = m_data[m_label]["ece"]

            # Diagonal perfect calibration line
            ax.plot([0, 1], [0, 1], color="#444444", linestyle="--", linewidth=1.0, alpha=0.8, zorder=2)

            width = 0.08
            for b in bins:
                if b["count"] > 0:
                    conf_c = b["confidence"]
                    acc_c = b["accuracy"]
                    ax.bar(
                        conf_c,
                        acc_c,
                        width=width,
                        color=method_colors[m_label],
                        alpha=0.65,
                        edgecolor="#222222",
                        linewidth=0.6,
                        zorder=3,
                    )
                    ax.plot([conf_c - width/2, conf_c + width/2], [conf_c, conf_c], color="#d62728", linewidth=1.5, zorder=4)

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_xticklabels(["0", "0.5", "1"])
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.set_yticklabels(["0", "0.5", "1"])
            ax.grid(True, linestyle=":", alpha=0.5, zorder=1)

            if row_idx == 0:
                ax.set_title(f"{m_label}", fontweight="bold", pad=3)
            if col_idx == 0:
                ax.set_ylabel(f"Qwen2.5-{m_name}\nAccuracy")
            if row_idx == 1:
                ax.set_xlabel("Confidence", labelpad=2)

            # ECE text badge
            ax.text(
                0.06, 0.80, f"ECE: {ece:.3f}",
                transform=ax.transAxes,
                fontsize=7.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#aaaaaa", alpha=0.9),
                zorder=5,
            )

    plt.tight_layout(pad=0.4)
    fig2_png = FIGURES_DIR / "fig2_reliability_diagrams.png"
    fig2_pdf = FIGURES_DIR / "fig2_reliability_diagrams.pdf"
    plt.savefig(fig2_png, dpi=300)
    plt.savefig(fig2_pdf)
    plt.close()
    print(f"Generated {fig2_png} and {fig2_pdf}")

    # -------------------------------------------------------------
    # Figure 3: ECE vs Model Size (Full 2-column width: 7.0 in)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for ax, (d_key, title) in zip(axes, datasets):
        sub_df = df[df["dataset"] == d_key]
        for meth_col, meth_label in METHOD_KEYS:
            meth_sub = sub_df[sub_df["method"] == meth_label]
            y_vals = []
            for slug, _, _ in MODEL_ORDER:
                row = meth_sub[meth_sub["model_slug"] == slug]
                y_vals.append(row["ece"].values[0] if not row.empty else np.nan)

            ax.plot(
                x_pos,
                y_vals,
                label=meth_label,
                color=method_colors[meth_label],
                marker=method_markers[meth_label],
                linewidth=1.8,
                markersize=5.5,
                zorder=3,
            )
        ax.set_title(title, fontweight="bold", fontsize=9.5, pad=3)
        ax.set_xlabel("Model Scale", fontsize=8.5, labelpad=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontweight="bold", fontsize=8.0)
        ax.set_xlim(-0.25, len(x_pos) - 0.75)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, linestyle=":", alpha=0.5, zorder=1)
        if ax == axes[0]:
            ax.set_ylabel("ECE (Calibration Error) ↓", fontsize=8.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cccccc",
        fontsize=7.5,
        handletextpad=0.3,
        columnspacing=1.2,
    )
    plt.subplots_adjust(top=0.83, bottom=0.17, left=0.08, right=0.98, wspace=0.12)
    fig3_png = FIGURES_DIR / "fig3_ece_vs_size.png"
    fig3_pdf = FIGURES_DIR / "fig3_ece_vs_size.pdf"
    plt.savefig(fig3_png, dpi=300)
    plt.savefig(fig3_pdf)
    plt.close()
    print(f"Generated {fig3_png} and {fig3_pdf}")

    # -------------------------------------------------------------
    # Figure 4: Verbalized Score Distribution (1-column 2x2 layout: 3.33 in)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(3.33, 2.3), sharey=True, sharex=True)
    axes_flat = axes.flatten()

    for idx, (ax, (slug, label, size)) in enumerate(zip(axes_flat, MODEL_ORDER)):
        records = raw_data.get(slug, [])
        correct_v = [float(r["conf_verbalized"]) * 100 for r in records if r["is_correct"]]
        wrong_v = [float(r["conf_verbalized"]) * 100 for r in records if not r["is_correct"]]

        ax.hist(
            [wrong_v, correct_v],
            bins=np.linspace(0, 100, 16),
            stacked=True,
            color=["#d62728", "#2ca02c"],
            label=["Wrong", "Correct"] if idx == 0 else None,
            edgecolor="#222222",
            linewidth=0.5,
            alpha=0.8,
            zorder=3,
        )
        ax.set_title(f"Qwen2.5-{label}", fontweight="bold", pad=2, fontsize=8.5)
        ax.set_xlim(-5, 105)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50", "100"])
        ax.set_ylim(0, 850)
        ax.set_yticks([0, 400, 800])
        ax.set_yticklabels(["0", "400", "800"])
        ax.grid(True, linestyle=":", alpha=0.5, zorder=1)
        if idx % 2 == 0:
            ax.set_ylabel("Count", fontsize=7.5)
        if idx >= 2:
            ax.set_xlabel("Score (0–100)", fontsize=7.5, labelpad=1)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7.0, framealpha=0.9, frameon=True, edgecolor="#cccccc")

    plt.tight_layout(pad=0.3)
    fig4_png = FIGURES_DIR / "fig4_verbalized_distribution.png"
    fig4_pdf = FIGURES_DIR / "fig4_verbalized_distribution.pdf"
    plt.savefig(fig4_png, dpi=300)
    plt.savefig(fig4_pdf)
    plt.close()
    print(f"Generated {fig4_png} and {fig4_pdf}")

    # Also generate a wide version of Fig 4 (1x4 grid, 7.0 in) for README
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.8), sharey=True)
    for idx, (ax, (slug, label, size)) in enumerate(zip(axes, MODEL_ORDER)):
        records = raw_data.get(slug, [])
        correct_v = [float(r["conf_verbalized"]) * 100 for r in records if r["is_correct"]]
        wrong_v = [float(r["conf_verbalized"]) * 100 for r in records if not r["is_correct"]]

        ax.hist(
            [wrong_v, correct_v],
            bins=np.linspace(0, 100, 16),
            stacked=True,
            color=["#d62728", "#2ca02c"],
            label=["Wrong", "Correct"] if idx == 0 else None,
            edgecolor="#222222",
            linewidth=0.5,
            alpha=0.8,
            zorder=3,
        )
        ax.set_title(f"Qwen2.5-{label}", fontweight="bold", pad=3, fontsize=9.0)
        ax.set_xlim(-5, 105)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50", "100"])
        ax.grid(True, linestyle=":", alpha=0.5, zorder=1)
        ax.set_xlabel("Verbalized (0–100)", fontsize=8.0)
        if idx == 0:
            ax.set_ylabel("Count", fontsize=8.0)
            ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9, frameon=True, edgecolor="#cccccc")

    plt.tight_layout(pad=0.4)
    fig4_wide_png = FIGURES_DIR / "fig4_verbalized_distribution_wide.png"
    fig4_wide_pdf = FIGURES_DIR / "fig4_verbalized_distribution_wide.pdf"
    plt.savefig(fig4_wide_png, dpi=300)
    plt.savefig(fig4_wide_pdf)
    plt.close()
    print(f"Generated {fig4_wide_png} and {fig4_wide_pdf}")


if __name__ == "__main__":
    run_full_analysis()
