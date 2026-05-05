"""Generate manuscript figures for Neural Router (Elsevier FGCS).

Output PDFs go to ../Manuscripts/Neural Router (Elsevier FGCS)/figs/.
All figures use matplotlib with deterministic styling — no GUI tools (L33).

Figures produced:
  fig:k-sensitivity            -> figs/k_sensitivity.pdf
  fig:tau-sensitivity          -> figs/tau_sensitivity.pdf
  fig:kappa-sensitivity        -> figs/kappa_sensitivity.pdf
  fig:embedding-sensitivity    -> figs/embedding_sensitivity.pdf
  fig:cost-validation          -> figs/cost_validation.pdf
  fig:scaling                  -> figs/scaling.pdf
  fig:pareto                   -> figs/pareto.pdf
  fig:d2-discrim (bar chart)   -> figs/d2_discrim.pdf
  fig:crossover                -> figs/crossover.pdf  (after data lands; skipped if missing)
  fig:qoe (Pareto)             -> figs/qoe_pareto.pdf (after data lands; skipped if missing)
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless, no GUI per L33
import matplotlib.pyplot as plt

# Paper-style defaults
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

import os
# Default output: ../figs/ relative to the experiments root (figs/ next
# to results/). Override via NROUTER_FIG_DIR env var when rendering into
# a manuscript directory.
OUT = Path(os.environ.get("NROUTER_FIG_DIR", Path(__file__).resolve().parent.parent / "figs"))
OUT.mkdir(parents=True, exist_ok=True)
SENS = Path("results/puhti_mirror/sensitivity_by_task")
SCALE = Path("results/puhti_mirror/scaling_by_task")

# ---------------------------------------------------------------------------
# fig:k-sensitivity — F1 and I vs k on D1 (Qwen-7B)
# ---------------------------------------------------------------------------
def fig_k_sensitivity():
    df = pd.read_csv(SENS / "qwen7b_sens_k_D1" / "sensitivity_k_D1.csv")
    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))
    ax2 = ax1.twinx()
    ax1.plot(df["k"], df["f1"], marker="o", color="C0", label="F1 (left)")
    ax1.set_xlabel("Number of clusters $k$")
    ax1.set_ylabel("Macro-F1", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_ylim(0, max(df["f1"].max() * 1.15, 0.5))
    ax2.plot(df["k"], df["invocations"], marker="s", color="C3", linestyle="--", label="$I$ (right)")
    ax2.set_ylabel("LLM invocations $I$", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.grid(False)
    plt.tight_layout()
    fig.savefig(OUT / "k_sensitivity.pdf")
    plt.close(fig)
    print(f"  saved k_sensitivity.pdf ({len(df)} k values)")

# ---------------------------------------------------------------------------
# fig:tau-sensitivity — F1, recall, I vs tau on D1
# ---------------------------------------------------------------------------
def fig_tau_sensitivity():
    df = pd.read_csv(SENS / "qwen7b_sens_tau_D1" / "sensitivity_tau_D1.csv")
    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))
    ax2 = ax1.twinx()
    ax1.plot(df["tau"], df["f1"], marker="o", color="C0", label="F1")
    ax1.plot(df["tau"], df["recall"], marker="^", color="C2", label="Recall", alpha=0.7)
    ax1.set_xlabel(r"Cosine threshold $\tau$")
    ax1.set_ylabel("Macro-F1 / Recall")
    ax1.set_ylim(0, max(df["f1"].max(), df["recall"].max()) * 1.15)
    ax1.legend(loc="upper right", fontsize=7)
    ax2.plot(df["tau"], df["invocations"], marker="s", color="C3", linestyle="--", label="$I$")
    ax2.set_ylabel("LLM invocations $I$", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.grid(False)
    plt.tight_layout()
    fig.savefig(OUT / "tau_sensitivity.pdf")
    plt.close(fig)
    print(f"  saved tau_sensitivity.pdf ({len(df)} tau values)")

# ---------------------------------------------------------------------------
# fig:kappa-sensitivity — F1, P, R vs kappa on D1
# ---------------------------------------------------------------------------
def fig_kappa_sensitivity():
    df = pd.read_csv(SENS / "qwen7b_sens_kappa_D1" / "sensitivity_kappa_D1.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(df["kappa"], df["f1"], marker="o", color="C0", label="F1")
    ax.plot(df["kappa"], df["precision"], marker="s", color="C1", label="Precision")
    ax.plot(df["kappa"], df["recall"], marker="^", color="C2", label="Recall")
    ax.set_xlabel(r"Top-$\kappa$ cut-off")
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(df[["f1","precision","recall"]].max().max() * 1.15, 0.5))
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT / "kappa_sensitivity.pdf")
    plt.close(fig)
    print(f"  saved kappa_sensitivity.pdf ({len(df)} kappa values)")

# ---------------------------------------------------------------------------
# fig:embedding-sensitivity — F1 grouped bar across embedding models
# ---------------------------------------------------------------------------
def fig_embedding_sensitivity():
    df = pd.read_csv(SENS / "qwen7b_sens_embedding_D1" / "sensitivity_embedding_D1.csv")
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    short_names = {
        "all-MiniLM-L6-v2": "MiniLM-L6",
        "all-mpnet-base-v2": "MPNet-base",
        "e5-large-v2": "E5-large",
        "bge-base-en-v1.5": "BGE-base",
    }
    df["short"] = df["embedding_model"].map(lambda s: short_names.get(s, s))
    bars = ax.bar(df["short"], df["f1"], color="C0", width=0.6)
    for b, v in zip(bars, df["f1"]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Embedding model")
    ax.set_ylabel("Macro-F1 (D1, A3)")
    ax.set_ylim(0, df["f1"].max() * 1.20)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(OUT / "embedding_sensitivity.pdf")
    plt.close(fig)
    print(f"  saved embedding_sensitivity.pdf ({len(df)} embeddings)")

# ---------------------------------------------------------------------------
# fig:cost-validation — per-cluster predicted vs measured I.
#
# Per L60: this figure justifies the §3.5/§4.6 empirical-validation claim
# for the cost model I = Σ_c ⌈m_c / b_max(c)⌉. Each marker is one cluster
# from the cost-validation re-run: we predict I_pred = ⌈m_c / b_max(|S_c|)⌉
# and compare against the measured per-cluster invocation count I_meas.
# Two backends: Qwen-2.5-7B (real LLM, Mahti GPU) + dry-run (laptop) — the
# latter exercises identical batching logic without GPU spend, providing
# additional grid coverage. Locked by tests/test_figure_data_consistency.py.
# ---------------------------------------------------------------------------
def fig_cost_validation():
    import json, math
    inputs = [
        ("Qwen-2.5-7B (Mahti)", "results/full/cost_validation/full/cost_validation_D1_qwen7b.csv"),
        ("dry-run (laptop)", "results/full/cost_validation/full/cost_validation_D1_dryrun.csv"),
    ]
    # Production cost-model constants from RouterConfig defaults (src/router.py).
    W, T_INST, T_RESP, T_S, T_E = 4096, 200, 500, 80, 30  # t_e estimated; t_s per Eq. 4
    rows = []
    for label, path in inputs:
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            try:
                per_inv = json.loads(r["per_cluster_invocations"])
                per_evt = json.loads(r["per_cluster_events"])
                per_sub = json.loads(r["per_cluster_active_subs"])
            except (TypeError, ValueError):
                continue
            for I_meas, m, S_c in zip(per_inv, per_evt, per_sub):
                if m == 0 or I_meas == 0:
                    continue  # cluster never invoked the LLM
                avail = W - T_INST - S_c * T_S - T_RESP
                if avail <= 0:
                    b_max = 1
                else:
                    b_max = max(1, math.floor(avail * 0.8 / max(T_E, 1)))
                I_pred = math.ceil(m / b_max)
                rows.append({"config": r["config"], "k": r["k"], "label": label,
                             "I_pred": I_pred, "I_meas": I_meas, "m": m, "S_c": S_c})
    if not rows:
        print("  cost_validation: NO DATA"); return
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    color_map = {"Qwen-2.5-7B (Mahti)": "C0", "dry-run (laptop)": "C7"}
    marker_map = {"Qwen-2.5-7B (Mahti)": "o", "dry-run (laptop)": "x"}
    for label in plot_df["label"].unique():
        sub = plot_df[plot_df["label"] == label]
        ax.scatter(sub["I_meas"], sub["I_pred"], color=color_map.get(label, "C2"),
                   marker=marker_map.get(label, "."), s=30, alpha=0.7,
                   label=f"{label} (n={len(sub)})", edgecolors="black", linewidth=0.3)
    lims = [0.8, max(plot_df[["I_pred","I_meas"]].max().max() * 1.5, 30)]
    ax.plot(lims, lims, "k--", alpha=0.6, linewidth=1, label="$I_{pred}=I_{meas}$")
    ax.fill_between(lims, [x*0.5 for x in lims], [x*2.0 for x in lims], alpha=0.12,
                    color="grey", label=r"factor-of-2 band")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Measured invocations $I_{\mathrm{meas}}$ per cluster")
    ax.set_ylabel(r"Predicted invocations $I_{\mathrm{pred}}$ per cluster")
    ax.legend(loc="lower right", fontsize=6.5)
    plt.tight_layout()
    fig.savefig(OUT / "cost_validation.pdf")
    plt.close(fig)
    print(f"  saved cost_validation.pdf ({len(plot_df)} per-cluster points)")

# ---------------------------------------------------------------------------
# fig:scaling — F1 + I vs |E| on D3 (Qwen-2.5-7B, A3) over the 50..2000 sweep
# Adds an analytical projection of I to |E| ∈ {5K, 10K, 50K, 100K} per §3.6.
# The empirical 5000-event point from qwen7b_scale_evt5000_D3 is intentionally
# excluded: its ~1.5 events/call batching does not match the ~5-7 events/call
# of the 50..2000 sweep, so combining them would conflate two regimes.
# Locked by tests/test_figure_data_consistency.py (per L60).
# ---------------------------------------------------------------------------
def fig_scaling():
    csv = SCALE / "qwen7b_scaling_events_D3" / "scaling_events_D3.csv"
    if not csv.exists():
        print("  scaling: NO DATA"); return
    df = pd.read_csv(csv).sort_values("n_events").drop_duplicates("n_events")
    fig, ax1 = plt.subplots(figsize=(3.8, 2.7))
    ax2 = ax1.twinx()
    # F1 (left axis): empirical only
    ax1.plot(df["n_events"], df["f1"], marker="o", color="C0", label="F1 (empirical)")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Number of events $|\mathcal{E}|$")
    ax1.set_ylabel("Macro-F1", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_ylim(0, max(df["f1"].max() * 1.2, 0.3))
    # I (right axis): empirical + analytical projection per §3.6
    ax2.plot(df["n_events"], df["invocations"], marker="s", color="C3", linestyle="--",
             label="$I$ (empirical)")
    coef = np.polyfit(df["n_events"].values, df["invocations"].values, 1)
    proj_n = np.array([5_000, 10_000, 50_000, 100_000])
    proj_I = np.polyval(coef, proj_n)
    ax2.plot(proj_n, proj_I, marker="^", color="C3", linestyle=":", alpha=0.6,
             label="$I$ (analytical)")
    ax2.set_yscale("log")
    ax2.set_ylabel("LLM invocations $I$", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper left", fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT / "scaling.pdf")
    plt.close(fig)
    print(f"  saved scaling.pdf ({len(df)} empirical + {len(proj_n)} analytical points)")

# ---------------------------------------------------------------------------
# fig:pareto — F1 vs invocations across backends + configs on D1.
# Invocations is a backend-agnostic resource proxy: each invocation incurs
# one prompt of comparable size (same config), so 'F1 vs I' is the honest
# accuracy-cost trade-off the manuscript §4.7 promises. No hardcoded event
# counts; no derived 'throughput' that conflated wall-clock with concurrency.
# Baselines have I=0 (no LLM); plotted at the left axis edge with a dummy x.
# ---------------------------------------------------------------------------
def fig_pareto():
    haiku = pd.read_csv("results/full/ablation/D1_ablation_haiku_results.csv")
    qwen = pd.read_csv("results/full/ablation/D1_ablation_qwen7b_per_task.csv")
    sonnet = pd.read_csv("results/full/ablation/D1_ablation_sonnet_results.csv")

    def agg(df, label):
        df = df[~df["config"].astype(str).str.startswith("baseline_")]
        g = df.groupby("config").agg(
            f1=("f1","mean"),
            inv=("invocations","mean"),
        ).reset_index()
        g["backend"] = label
        return g

    h = agg(haiku, "Haiku")
    q = agg(qwen, "Qwen-2.5-7B")
    s = agg(sonnet, "Sonnet (1 seed)")
    df = pd.concat([h, q, s], ignore_index=True)

    bl = pd.read_csv("results/full/ablation/D1_ablation_haiku_results.csv")
    bl = bl[bl["config"].astype(str).str.startswith("baseline_")].groupby("config").agg(
        f1=("f1","mean")
    ).reset_index()
    # Baselines do not invoke an LLM. Plot at I=1 on the log axis so they sit
    # at the far left rather than blowing up log(0).
    bl["inv"] = 1
    bl["backend"] = "Baselines (I=0)"

    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    color_map = {"Haiku": "C1", "Qwen-2.5-7B": "C0", "Sonnet (1 seed)": "C3",
                 "Baselines (I=0)": "C7"}
    marker_map = {"Haiku": "o", "Qwen-2.5-7B": "s", "Sonnet (1 seed)": "^",
                  "Baselines (I=0)": "x"}
    for backend in ["Haiku", "Qwen-2.5-7B", "Sonnet (1 seed)"]:
        sub = df[df["backend"] == backend]
        ax.scatter(sub["inv"], sub["f1"], color=color_map[backend], marker=marker_map[backend],
                   s=42, label=backend, alpha=0.85, edgecolors="black", linewidth=0.3)
        # Annotate each point with its config name for readability
        for _, row in sub.iterrows():
            ax.annotate(row["config"], (row["inv"], row["f1"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=6, alpha=0.7)
    ax.scatter(bl["inv"], bl["f1"], color=color_map["Baselines (I=0)"],
               marker=marker_map["Baselines (I=0)"], s=42, label="Baselines (I=0)", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel(r"LLM invocations $I$ per run (log; baselines at $I{=}1$)")
    ax.set_ylabel("Macro-F1 (D1)")
    full = pd.concat([df[["f1","inv"]], bl[["f1","inv"]]], ignore_index=True)
    ax.set_ylim(0, full["f1"].max() * 1.15)
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT / "pareto.pdf")
    plt.close(fig)
    print(f"  saved pareto.pdf ({len(df) + len(bl)} (config, backend) cells)")

# ---------------------------------------------------------------------------
# fig:d2-discrim — discrimination-capacity panel as bar chart
# ---------------------------------------------------------------------------
def fig_d2_discrim():
    # Hand-coded panel data because tab:d2-discrim aggregates from 3 sources.
    panel = pd.DataFrame([
        {"backend": "Qwen-2.5\n1.5B",  "params": 1.5,  "A0": 0.000, "A1": 0.001, "kind": "open-weight"},
        {"backend": "Qwen-2.5\n7B",    "params": 7.0,  "A0": 0.002, "A1": 0.045, "kind": "open-weight"},
        {"backend": "Mistral\n7B",     "params": 7.0,  "A0": 0.039, "A1": 0.080, "kind": "open-weight"},
        {"backend": "Qwen-2.5\n32B",   "params": 32.0, "A0": 0.127, "A1": 0.116, "kind": "open-weight"},
        {"backend": "Sonnet\n(5K, 1 seed)", "params": np.nan, "A0": 0.316, "A1": np.nan, "kind": "closed-API"},
    ])
    sbert = 0.154
    tfidf = 0.162

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    x = np.arange(len(panel))
    width = 0.35
    a0 = ax.bar(x - width/2, panel["A0"], width, color="C0", label="A0 (raw LLM)", edgecolor="black", linewidth=0.3)
    a1 = ax.bar(x + width/2, panel["A1"].fillna(0), width, color="C1", label="A1 (clustering)", edgecolor="black", linewidth=0.3)
    # value labels
    for b, v in zip(a0, panel["A0"]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    for b, v in zip(a1, panel["A1"]):
        if pd.notna(v):
            ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    ax.axhline(sbert, color="C2", linestyle="--", linewidth=1, label=f"SBERT cosine ({sbert:.3f})")
    ax.axhline(tfidf, color="C3", linestyle=":", linewidth=1, label=f"TF-IDF cosine ({tfidf:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(panel["backend"])
    ax.set_ylabel("Macro-F1 on D2 ($|\mathcal{S}|=201$)")
    ax.set_ylim(0, max(panel[["A0","A1"]].max().max(), 0.35) * 1.15)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(OUT / "d2_discrim.pdf")
    plt.close(fig)
    print("  saved d2_discrim.pdf (5-backend panel)")

# ---------------------------------------------------------------------------
# fig:crossover — A0 (truncates) vs A4 (compresses then truncates) under the
# enforced W=4096 budget on D1 across |S| ∈ {50, 100, 200, 500, 1000, 2000}.
# Solid lines = F1; dashed = per-event token cost. Two y-axes; vertical
# line at the analytical |S|·t_s = W threshold (≈ (W - t_inst - t_resp) / t_s
# = (4096 - 200 - 500) / 80 ≈ 42 subscriptions for default token estimates).
# ---------------------------------------------------------------------------
def fig_crossover():
    """Render the empirical crossover sweep, if its data files exist.

    Prefers the description-aware results (crossover_desc/) over the
    legacy ID-based results that were retired after the L61 metric-
    artifact diagnosis (2026-05-04). Falls back to the legacy dirs only
    if no desc-aware data has landed yet.
    """
    candidates = [
        Path("results/full/crossover_desc/by_task"),
        Path("results/mahti_mirror/crossover/by_task"),
        Path("results/full/crossover/by_task"),
    ]
    by_task = next((p for p in candidates if p.exists()), None)
    if by_task is None:
        print("  fig:crossover SKIPPED — no by_task dir found (Mahti results not pulled yet)")
        return

    rows = []
    for d in sorted(by_task.iterdir()):
        if not d.name.startswith("qwen7b_cross_"): continue
        # Expect a single CSV per dir from run_crossover.py
        for c in d.glob("*.csv"):
            try:
                df_ = pd.read_csv(c)
                rows.append(df_)
            except Exception:
                continue
    if not rows:
        print("  fig:crossover SKIPPED — no CSV rows found in by_task/")
        return

    df = pd.concat(rows, ignore_index=True)
    if df.empty or "n_subscriptions" not in df.columns:
        print("  fig:crossover SKIPPED — empty/malformed data")
        return

    # Aggregate across seeds: mean F1 per (config, n_subscriptions)
    agg = df.groupby(["config", "n_subscriptions"]).agg(
        f1_mean=("f1", "mean"),
        cost_mean=("cost_per_1k", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(4.0, 2.8))
    ax2 = ax1.twinx()
    colors = {"A0": "C0", "A4": "C1"}
    markers = {"A0": "o", "A4": "s"}
    for cfg in ["A0", "A4"]:
        sub = agg[agg["config"] == cfg].sort_values("n_subscriptions")
        if sub.empty: continue
        ax1.plot(sub["n_subscriptions"], sub["f1_mean"], marker=markers[cfg],
                 color=colors[cfg], label=f"{cfg} F1")
        ax2.plot(sub["n_subscriptions"], sub["cost_mean"], marker=markers[cfg],
                 color=colors[cfg], linestyle="--", alpha=0.5,
                 label=f"{cfg} cost (\\$/1k evt)")

    # Analytical threshold: |S| · t_s = W - t_inst - t_resp
    # = (4096 - 200 - 500) / 80 = ~42 subscriptions
    threshold = (4096 - 200 - 500) / 80
    ax1.axvline(threshold, color="k", linestyle=":", alpha=0.5,
                label=f"|S|·t_s = W ({threshold:.0f})")

    ax1.set_xscale("log")
    ax1.set_xlabel("Subscription volume |S|")
    ax1.set_ylabel("Macro-F1", color="C0")
    ax2.set_ylabel("Per-event token cost ($/1k events)")
    ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=6.5, ncol=2)
    plt.tight_layout()
    fig.savefig(OUT / "crossover.pdf")
    plt.close(fig)
    print(f"  saved crossover.pdf ({len(agg)} (config, |S|) cells)")


# ---------------------------------------------------------------------------
# fig:qoe — Pareto F1 vs cost across QoE strategies on D1 with three
# Qwen tiers (1.5B/7B/32B). Colour by strategy, shape by weight preset.
# ---------------------------------------------------------------------------
def fig_qoe():
    """Render the QoE Pareto chart, if its data file exists."""
    candidates = [
        Path("results/mahti_mirror/qoe/by_task/qwen-tiers_D1/qoe_D1.csv"),
        Path("results/full/qoe/by_task/qwen-tiers_D1/qoe_D1.csv"),
    ]
    csv = next((c for c in candidates if c.exists()), None)
    if csv is None:
        print("  fig:qoe SKIPPED — no QoE CSV found")
        return
    df = pd.read_csv(csv)
    if df.empty:
        print("  fig:qoe SKIPPED — empty CSV")
        return

    # Aggregate per (strategy, weight_preset, backend): mean F1, mean cost, mean L
    agg = df.groupby(["strategy", "weight_preset", "backend"]).agg(
        f1=("f1", "mean"),
        cost=("cost_per_1k", "mean"),
        latency=("latency_s", "mean"),
        n=("seed", "nunique"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    strategy_color = {"homogeneous": "C0", "round_robin": "C2", "qoe_optimised": "C3"}
    preset_marker = {"n/a": "x", "accuracy_first": "^", "balanced": "o", "cost_first": "v"}
    for _, row in agg.iterrows():
        c = strategy_color.get(row["strategy"], "C7")
        m = preset_marker.get(row["weight_preset"], "o")
        ax.scatter(row["cost"], row["f1"], color=c, marker=m, s=50, alpha=0.85,
                   edgecolors="black", linewidth=0.4)
    # Build a deduplicated legend manually
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    legend_handles = [
        mpatches.Patch(color=strategy_color["homogeneous"], label="homogeneous"),
        mpatches.Patch(color=strategy_color["round_robin"], label="round-robin"),
        mpatches.Patch(color=strategy_color["qoe_optimised"], label="QoE-optimised"),
    ]
    legend_handles += [
        mlines.Line2D([], [], color="black", marker=preset_marker["accuracy_first"],
                      linestyle="None", markersize=6, label="accuracy-first"),
        mlines.Line2D([], [], color="black", marker=preset_marker["balanced"],
                      linestyle="None", markersize=6, label="balanced"),
        mlines.Line2D([], [], color="black", marker=preset_marker["cost_first"],
                      linestyle="None", markersize=6, label="cost-first"),
    ]
    ax.set_xlabel(r"Cost (\$/1k events)")
    ax.set_ylabel("Macro-F1 (D1)")
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(OUT / "qoe_pareto.pdf")
    plt.close(fig)
    print(f"  saved qoe_pareto.pdf ({len(agg)} (strategy, preset, backend) cells)")


if __name__ == "__main__":
    print(f"Output dir: {OUT}")
    fig_k_sensitivity()
    fig_tau_sensitivity()
    fig_kappa_sensitivity()
    fig_embedding_sensitivity()
    fig_cost_validation()
    fig_scaling()
    fig_pareto()
    fig_d2_discrim()
    fig_crossover()
    fig_qoe()
    print("done.")
