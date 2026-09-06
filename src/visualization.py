"""
visualization.py
================
EDA plotting functions extracted from notebooks/01_EDA.ipynb.
Each function saves its figure into results/figures/ and shows it inline
so it works fine when called from a notebook too.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import squarify

from src.config import RESULTS_FIGURES_DIR


def _savefig(filename: str) -> None:
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()


def plot_missing(df, save: bool = True) -> None:
    """Heatmap of missing values across all columns/rows."""
    plt.figure(figsize=(14, 6))
    ax = plt.axes()
    sns.heatmap(df.isna().transpose(), cbar=False, ax=ax)
    ax.set_title("Missing Values Heatmap", fontsize=14, fontweight="bold")
    plt.xlabel("Rows")
    plt.ylabel("Columns")
    if save:
        _savefig("01_missing_values_heatmap.png")
    else:
        plt.show()


def plot_correlation_heatmap(df, columns, method: str = "spearman", save: bool = True) -> None:
    """Correlation heatmap for the given numeric columns."""
    correlation_matrix = df[columns].corr(method=method)

    sns.set_theme(style="dark")
    plt.figure(figsize=(10, 10))
    heatmap = sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="PuBu",
        linewidths=0.3,
        linecolor="black",
        annot_kws={"size": 18, "weight": "bold"},
        cbar_kws={"label": f"{method.capitalize()} Correlation", "shrink": 0.8},
    )
    heatmap.set_title(f"Correlation Heatmap ({method.capitalize()})", fontsize=14, fontweight="bold")
    if save:
        _savefig("02_correlation_heatmap.png")
    else:
        plt.show()


def plot_categories_treemap(category_col, top_n: int = 100, group_size: int = 10, save: bool = True) -> None:
    """Treemap of the most common book categories."""
    top = category_col.value_counts().head(top_n)

    n_groups = max(top_n // group_size, 1)
    blue_shades = plt.cm.Blues(np.linspace(1, 0.2, n_groups))

    colors = []
    for i in range(len(top)):
        group_idx = min(i // group_size, n_groups - 1)
        colors.append(blue_shades[group_idx])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [10, 1]})
    squarify.plot(
        sizes=top.values,
        label=[f"{val}" for _, val in zip(top.index, top.values)],
        color=colors,
        alpha=0.9,
        text_kwargs={"fontsize": 8, "color": "white", "fontweight": "bold"},
        ax=ax1,
    )
    ax1.set_title("Categories Distribution", fontsize=18, fontweight="bold", pad=20)
    ax1.axis("off")

    legend_patches = [
        mpatches.Patch(color=blue_shades[i], label=f"Rank {i * group_size + 1}\u2013{min((i + 1) * group_size, len(top))}")
        for i in range(n_groups)
    ]
    ax2.legend(handles=legend_patches, loc="center", ncol=n_groups, frameon=False, fontsize=9)
    ax2.axis("off")

    if save:
        _savefig("03_categories_treemap.png")
    else:
        plt.tight_layout()
        plt.show()


def plot_top_categories(category_col, top_n: int, save: bool = True) -> None:
    """Horizontal bar chart of the top N book categories."""
    top = category_col.value_counts().head(top_n)
    colors = plt.cm.Blues(np.linspace(0.4, 1, top_n))
    labels = [f"{top_n - i}. {cat}" for i, cat in enumerate(top.index[::-1])]

    fig, ax = plt.subplots(figsize=(14, top_n * 0.5))
    bars = ax.barh(labels, top.values[::-1], color=colors)

    for bar, val in zip(bars, top.values[::-1]):
        ax.text(
            bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", ha="center",
            fontsize=10, fontweight="bold", color="white",
        )

    ax.set_title(f"Top {top_n} Categories", fontsize=14, fontweight="bold")
    if save:
        _savefig("04_top_categories.png")
    else:
        plt.tight_layout()
        plt.show()


def plot_histogram(df_col, bins: int = 10, bin_per_value: bool = False, title: str = "Distribution", save: bool = True, filename: str = "05_histogram.png") -> None:
    """Generic histogram with mean/median/std reference lines."""
    if bin_per_value:
        bins = int(df_col.dropna().max() - df_col.dropna().min())

    data = df_col.dropna()
    mean = data.mean()
    median = data.median()
    std = data.std()

    plt.figure(figsize=(14, 6))
    plt.hist(data, bins=bins, color="#1a6faf", edgecolor="#0d3d61")

    plt.axvline(mean, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean:.1f}")
    plt.axvline(median, color="orange", linestyle="--", linewidth=1.5, label=f"Median: {median:.1f}")
    plt.axvline(mean + std, color="green", linestyle=":", linewidth=1, label=f"Std: \u00b1{std:.1f}")
    plt.axvline(mean - std, color="green", linestyle=":", linewidth=1)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.legend(fontsize=10)

    if save:
        _savefig(filename)
    else:
        plt.tight_layout()
        plt.show()
