#!/usr/bin/env python3
"""
Visualize experiments MLP one-step experiment results (experiments_mlp.json).

Usage:
  uv run python scripts/visualize_experiments_mlp.py
  uv run python scripts/visualize_experiments_mlp.py --input results/experiments_mlp.json --output-dir results/experiments_plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "results" / "experiments_mlp.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "experiments_plots"

CHECKPOINT_ORDER = ["v1", "v2", "v3"]
CHECKPOINT_LABELS = {
    "v1": "V1 (Retro* default)",
    "v2": "V2 (value ours)",
    "v3": "V3 (zero ours)",
}
COLORS = {"v1": "#4C72B0", "v2": "#DD8452", "v3": "#55A868"}


def load_results(path: Path) -> tuple[pd.DataFrame, dict]:
    with path.open() as f:
        payload = json.load(f)
    df = pd.DataFrame(payload["summary"])
    df = df[df["n_predictions"] > 0].copy()
    return df, payload


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ck, topk), grp in df.groupby(["checkpoint", "topk"], sort=False):
        hits = grp["hit"].fillna(False).astype(bool)
        hit_ranks = grp.loc[hits, "hit_rank"].dropna()
        rows.append(
            {
                "checkpoint": ck,
                "topk": int(topk),
                "n_molecules": len(grp),
                "hit_rate": hits.mean(),
                "mean_hit_rank": hit_ranks.mean() if len(hit_ranks) else np.nan,
                "mean_frac_precursors_available": grp["frac_precursors_available"].mean(),
                "mean_frac_reactions_all_available": grp[
                    "frac_reactions_all_available"
                ].mean(),
                "mean_inference_s": grp["inference_s"].mean(),
            }
        )
    out = pd.DataFrame(rows)
    out["checkpoint"] = out["checkpoint"].astype(str)
    ck_order = [c for c in CHECKPOINT_ORDER if c in out["checkpoint"].unique()]
    out["checkpoint"] = pd.Categorical(out["checkpoint"], categories=ck_order, ordered=True)
    return out.sort_values(["checkpoint", "topk"])


def plot_hit_rate_by_topk(agg: pd.DataFrame, out_dir: Path) -> Path:
    checkpoints = list(agg["checkpoint"].cat.categories)
    topks = sorted(agg["topk"].unique())
    x = np.arange(len(topks))
    width = 0.35 if len(checkpoints) == 2 else 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, ck in enumerate(checkpoints):
        sub = agg[agg["checkpoint"] == ck].set_index("topk")
        vals = [sub.loc[k, "hit_rate"] * 100 if k in sub.index else 0 for k in topks]
        offset = (i - (len(checkpoints) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=CHECKPOINT_LABELS.get(ck, ck),
            color=COLORS.get(ck, f"C{i}"),
            edgecolor="white",
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"top-{k}" for k in topks])
    ax.set_ylabel("Hit rate (%)")
    ax.set_xlabel("Top-k predictions")
    ax.set_ylim(0, 105)
    ax.set_title("Ground-truth hit@k (first retrosynthetic step)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "01_hit_rate_by_topk.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_per_molecule_heatmap(df: pd.DataFrame, topk: int, out_dir: Path) -> Path:
    sub = df[df["topk"] == topk].copy()
    checkpoints = [
        c for c in CHECKPOINT_ORDER if c in sub["checkpoint"].unique()
    ]
    routes = sorted(sub["route_index"].unique())

    hit_matrix = np.full((len(checkpoints), len(routes)), np.nan)
    rank_matrix = np.full((len(checkpoints), len(routes)), np.nan)

    for i, ck in enumerate(checkpoints):
        for j, route in enumerate(routes):
            row = sub[(sub["checkpoint"] == ck) & (sub["route_index"] == route)]
            if row.empty:
                continue
            r = row.iloc[0]
            hit_matrix[i, j] = 1.0 if r["hit"] else 0.0
            if r["hit"] and pd.notna(r["hit_rank"]):
                rank_matrix[i, j] = float(r["hit_rank"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5), gridspec_kw={"width_ratios": [1, 1.2]})

    im0 = axes[0].imshow(hit_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axes[0].set_yticks(range(len(checkpoints)))
    axes[0].set_yticklabels([CHECKPOINT_LABELS.get(c, c) for c in checkpoints])
    axes[0].set_xticks(range(len(routes)))
    axes[0].set_xticklabels([f"R{r}" for r in routes], rotation=45, ha="right")
    axes[0].set_title(f"Hit / miss (top-{topk})")
    for i in range(hit_matrix.shape[0]):
        for j in range(hit_matrix.shape[1]):
            if np.isnan(hit_matrix[i, j]):
                continue
            axes[0].text(j, i, "✓" if hit_matrix[i, j] else "✗", ha="center", va="center", fontsize=12)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="hit")

    masked = np.ma.masked_where(np.isnan(rank_matrix) | (hit_matrix == 0), rank_matrix)
    im1 = axes[1].imshow(masked, aspect="auto", cmap="viridis_r", vmin=0, vmax=max(topk - 1, 1))
    axes[1].set_yticks(range(len(checkpoints)))
    axes[1].set_yticklabels([CHECKPOINT_LABELS.get(c, c) for c in checkpoints])
    axes[1].set_xticks(range(len(routes)))
    axes[1].set_xticklabels([f"R{r}" for r in routes], rotation=45, ha="right")
    axes[1].set_title(f"Hit rank when found (0 = best, top-{topk})")
    for i in range(rank_matrix.shape[0]):
        for j in range(rank_matrix.shape[1]):
            if np.isnan(rank_matrix[i, j]) or hit_matrix[i, j] == 0:
                continue
            axes[1].text(j, i, f"{int(rank_matrix[i, j])}", ha="center", va="center", color="white", fontsize=10)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="rank")

    fig.suptitle(f"Per-route comparison at top-{topk}", y=1.02)
    fig.tight_layout()
    path = out_dir / f"02_per_route_top{topk}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_stock_metrics(agg: pd.DataFrame, out_dir: Path) -> Path:
    checkpoints = list(agg["checkpoint"].cat.categories)
    topks = sorted(agg["topk"].unique())
    x = np.arange(len(topks))
    width = 0.35 if len(checkpoints) == 2 else 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, ck in enumerate(checkpoints):
        sub = agg[agg["checkpoint"] == ck].set_index("topk")
        vals = [
            sub.loc[k, "mean_frac_precursors_available"] * 100 if k in sub.index else 0
            for k in topks
        ]
        offset = (i - (len(checkpoints) - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=CHECKPOINT_LABELS.get(ck, ck),
            color=COLORS.get(ck, f"C{i}"),
            alpha=0.85,
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"top-{k}" for k in topks])
    ax.set_ylabel("Mean fraction available (%)")
    ax.set_xlabel("Top-k predictions")
    ax.set_title("Precursor availability in top-k (origin_dict stock)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "03_precursor_availability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_hit_rank_distribution(df: pd.DataFrame, topk: int, out_dir: Path) -> Path:
    sub = df[(df["topk"] == topk) & (df["hit"] == True)].copy()  # noqa: E712
    if sub.empty:
        return out_dir / f"04_hit_rank_dist_top{topk}.png"

    checkpoints = [
        c for c in CHECKPOINT_ORDER if c in sub["checkpoint"].unique()
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(-0.5, topk + 0.5, 1)

    for ck in checkpoints:
        ranks = sub.loc[sub["checkpoint"] == ck, "hit_rank"].dropna().astype(int)
        ax.hist(
            ranks,
            bins=bins,
            alpha=0.55,
            label=CHECKPOINT_LABELS.get(ck, ck),
            color=COLORS.get(ck, None),
            edgecolor="white",
        )

    ax.set_xticks(range(topk))
    ax.set_xlabel("Hit rank (0 = top prediction)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of hit ranks (hits only, top-{topk})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"04_hit_rank_dist_top{topk}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_inference_time(agg: pd.DataFrame, out_dir: Path) -> Path:
    checkpoints = list(agg["checkpoint"].cat.categories)
    topks = sorted(agg["topk"].unique())

    fig, ax = plt.subplots(figsize=(8, 4))
    for ck in checkpoints:
        sub = agg[agg["checkpoint"] == ck].sort_values("topk")
        ax.plot(
            sub["topk"],
            sub["mean_inference_s"],
            marker="o",
            label=CHECKPOINT_LABELS.get(ck, ck),
            color=COLORS.get(ck, None),
        )

    ax.set_xlabel("Top-k")
    ax.set_ylabel("Mean inference time (s)")
    ax.set_title("MLP inference latency vs top-k")
    ax.set_xticks(topks)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "05_inference_time.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_text_summary(df: pd.DataFrame, agg: pd.DataFrame, payload: dict) -> None:
    n_routes = df["route_index"].nunique()
    checkpoints = sorted(df["checkpoint"].unique())
    topks = sorted(df["topk"].unique())

    print("=" * 72)
    print("Experiments MLP — text summary")
    print("=" * 72)
    print(f"Molecules: {n_routes}  |  Checkpoints: {', '.join(checkpoints)}  |  top-k: {topks}")
    print()

    for _, row in agg.iterrows():
        ck = str(row["checkpoint"])
        print(
            f"  {CHECKPOINT_LABELS.get(ck, ck):22s}  top-{int(row['topk']):2d}  "
            f"hit@{int(row['topk'])}={row['hit_rate']:.1%}  "
            f"mean rank={row['mean_hit_rank']:.2f}  "
            f"precursors avail={row['mean_frac_precursors_available']:.1%}  "
            f"inference={row['mean_inference_s']:.2f}s"
        )

    print()
    print("Interpretation notes:")
    print(f"  • Sample size n={n_routes} — pilot only; do not treat V1 vs V2 differences as significant.")
    print("  • hit@k: GT first step found in top-k canonical reactant sets.")
    print("  • rxns_all_avail ≈ 0% means no predicted reaction has all precursors in stock.")
    print("  • V1 uses internal topk=50 then truncates; V2 uses requested k directly.")

    if "aggregate_hit_rate" in payload:
        print()
        print("Precomputed aggregate_hit_rate from JSON:")
        for key, val in payload["aggregate_hit_rate"].items():
            if val is not None:
                print(f"  {key}: {val:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize experiments_mlp.json results")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--heatmap-topk",
        type=int,
        default=10,
        help="top-k for per-route heatmap (default: 10)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Results not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, payload = load_results(args.input)
    agg = aggregate_metrics(df)
    agg.to_csv(args.output_dir / "aggregated_metrics.csv", index=False)

    print_text_summary(df, agg, payload)

    paths = [
        plot_hit_rate_by_topk(agg, args.output_dir),
        plot_per_molecule_heatmap(df, args.heatmap_topk, args.output_dir),
        plot_stock_metrics(agg, args.output_dir),
        plot_hit_rank_distribution(df, args.heatmap_topk, args.output_dir),
        plot_inference_time(agg, args.output_dir),
    ]

    print()
    print("Saved plots:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
