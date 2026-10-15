#!/usr/bin/env python
"""Major-revision supplementary experiments for Solderbench (SoftwareX).

Produces, under results/revision/ and results/figures/:
  * table_baseline_multiseed.{csv,tex}  -- 5 independent seeds, mean +/- std
  * table_noise_ablation.{csv,tex}      -- sensor-noise robustness
  * table_defect_ablation.{csv,tex}     -- defect-strength / overlap robustness
  * sample_clouds.png                   -- one representative cloud per class
  * noise_effect.png                    -- noise effect on a nominal joint
  * defect_effect.png                   -- defect-strength effect on insufficient
  * summary.json                        -- all numbers consumed by the paper edit
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solderbench import load_dataset, run_comparison, SolderJointGenerator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "revision")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1011]
N = 250
NOISE = 0.005  # matches run_benchmark.py default -> reproduces paper Table 3 (RF 0.9440)
NPOINTS = 1500


def _cell(m, s):
    return f"{m:.4f}$\\pm${s:.4f}"


# ----------------------------------------------------------------------
# A) Multi-seed baseline comparison
# ----------------------------------------------------------------------
print("A) multi-seed baseline ...")
records = {s: {} for s in SEEDS}
for seed in SEEDS:
    X, y = load_dataset(n_per_class=N, seed=seed, noise=NOISE, n_points=NPOINTS)
    df = run_comparison(X, y, seed=seed)
    for _, r in df.iterrows():
        records[seed][r["no"]] = {
            "name": r["name"], "Accuracy": r["Accuracy"],
            "Precision": r["Precision"], "Recall": r["Recall"],
            "F1": r["F1"], "CV_mean": r["CV_mean"], "CV_std": r["CV_std"],
        }

rows = []
for no in ["B1", "B2", "B3", "B4", "B5", "B6"]:
    acc = np.array([records[s][no]["Accuracy"] for s in SEEDS])
    pre = np.array([records[s][no]["Precision"] for s in SEEDS])
    rec = np.array([records[s][no]["Recall"] for s in SEEDS])
    f1 = np.array([records[s][no]["F1"] for s in SEEDS])
    cv = np.array([records[s][no]["CV_mean"] for s in SEEDS])
    cvs = np.array([records[s][no]["CV_std"] for s in SEEDS])
    rows.append({
        "no": no, "name": records[SEEDS[0]][no]["name"],
        "Acc_mean": acc.mean(), "Acc_std": acc.std(),
        "P_mean": pre.mean(), "P_std": pre.std(),
        "R_mean": rec.mean(), "R_std": rec.std(),
        "F1_mean": f1.mean(), "F1_std": f1.std(),
        "CV_mean": cv.mean(), "CV_std": cvs.mean(),
    })
multi = pd.DataFrame(rows)
multi.to_csv(os.path.join(OUT, "table_baseline_multiseed.csv"), index=False)

lines = [
    r"\begin{table}[htbp]", r"  \centering",
    r"  \caption{Baseline comparison across five independent random seeds "
    r"(mean $\pm$ std; 1250 samples per seed, stratified 7:3 split, 5-fold CV on "
    r"the training set). Best mean in bold.}",
    r"  \label{tab:baseline_multi}",
    r"  \begin{tabular}{lcccccc}", r"    \toprule",
    r"    No. & Method & Accuracy & Precision & Recall & macro-$F_1$ & CV Acc. \\",
    r"    \midrule",
]
for _, r in multi.iterrows():
    ours = "Ours" in r["name"]
    name = r["name"].replace(" (Ours)", "")
    if ours:
        name = r"\textbf{" + name + " (Ours)}"
    lines.append(
        f"    {r['no']} & {name} & {_cell(r['Acc_mean'], r['Acc_std'])} & "
        f"{_cell(r['P_mean'], r['P_std'])} & {_cell(r['R_mean'], r['R_std'])} & "
        f"{_cell(r['F1_mean'], r['F1_std'])} & {_cell(r['CV_mean'], r['CV_std'])} \\\\"
    )
lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
with open(os.path.join(OUT, "table_baseline_multiseed.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

rf = np.array([records[s]["B6"]["Accuracy"] for s in SEEDS])
gb = np.array([records[s]["B5"]["Accuracy"] for s in SEEDS])
gap = rf - gb
gap_mean, gap_std = float(gap.mean()), float(gap.std())
print(f"   RF vs GB accuracy gap across seeds: mean={gap_mean:+.4f} std={gap_std:.4f}")

# ----------------------------------------------------------------------
# B) Sensor-noise ablation
# ----------------------------------------------------------------------
print("B) noise ablation ...")
noise_levels = [0.005, 0.01, 0.02, 0.05, 0.10]
nrows = []
for nl in noise_levels:
    X, y = load_dataset(n_per_class=N, seed=42, noise=nl, n_points=NPOINTS)
    df = run_comparison(X, y, seed=42)
    d = {r["no"]: r for _, r in df.iterrows()}
    nrows.append({
        "noise": nl,
        "RF_Acc": d["B6"]["Accuracy"], "RF_F1": d["B6"]["F1"],
        "S3DA_Acc": d["B1"]["Accuracy"], "GB_Acc": d["B5"]["Accuracy"],
        "min_Acc": df["Accuracy"].min(), "max_Acc": df["Accuracy"].max(),
    })
noise_df = pd.DataFrame(nrows)
noise_df.to_csv(os.path.join(OUT, "table_noise_ablation.csv"), index=False)

lines = [
    r"\begin{table}[htbp]", r"  \centering",
    r"  \caption{Robustness to sensor noise. Accuracy and macro-$F_1$ of the "
    r"proposed random forest (B6) and the S3DA volume-threshold rule (B1) as a "
    r"function of the Gaussian noise standard deviation $\sigma$ (seed 42).}",
    r"  \label{tab:noise}",
    r"  \begin{tabular}{lcccc}", r"    \toprule",
    r"    $\sigma$ & RF Acc. & RF macro-$F_1$ & S3DA Acc. & SpAN$_{\mathrm{acc}}$ \\",
    r"    \midrule",
]
for _, r in noise_df.iterrows():
    lines.append(
        f"    {r['noise']:.3f} & {r['RF_Acc']:.4f} & {r['RF_F1']:.4f} & "
        f"{r['S3DA_Acc']:.4f} & [{r['min_Acc']:.4f}, {r['max_Acc']:.4f}] \\\\"
    )
lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
with open(os.path.join(OUT, "table_noise_ablation.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# ----------------------------------------------------------------------
# C) Defect-strength / class-overlap ablation
# ----------------------------------------------------------------------
print("C) defect-strength ablation ...")
configs = {
    "narrow (easier)": dict(
        insufficient_scale_range=(0.62, 0.80), excess_scale_std=0.06,
        collapse_depth_range=(0.30, 0.42), bridge_dist_std=0.03,
        excess_bump_range=(150, 250)),
    "default": dict(),
    "wide (harder)": dict(
        insufficient_scale_range=(0.45, 0.92), excess_scale_std=0.18,
        collapse_depth_range=(0.25, 0.55), bridge_dist_std=0.09,
        excess_bump_range=(200, 400)),
}
drows = []
for cname, cfg in configs.items():
    X, y = load_dataset(n_per_class=N, seed=42, noise=NOISE, n_points=NPOINTS, **cfg)
    df = run_comparison(X, y, seed=42)
    d = {r["no"]: r for _, r in df.iterrows()}
    drows.append({
        "config": cname,
        "RF_Acc": d["B6"]["Accuracy"], "RF_F1": d["B6"]["F1"],
        "S3DA_Acc": d["B1"]["Accuracy"], "min_Acc": df["Accuracy"].min(),
    })
defect_df = pd.DataFrame(drows)
defect_df.to_csv(os.path.join(OUT, "table_defect_ablation.csv"), index=False)

lines = [
    r"\begin{table}[htbp]", r"  \centering",
    r"  \caption{Robustness to defect-strength / class-overlap configuration. "
    r"RF accuracy and macro-$F_1$ under a narrow (easier), default, and wide "
    r"(harder) overlap of the class parameter ranges (seed 42).}",
    r"  \label{tab:defect}",
    r"  \begin{tabular}{lcccc}", r"    \toprule",
    r"    Configuration & RF Acc. & RF macro-$F_1$ & S3DA Acc. & Min acc. \\",
    r"    \midrule",
]
for _, r in defect_df.iterrows():
    lines.append(
        f"    {r['config']} & {r['RF_Acc']:.4f} & {r['RF_F1']:.4f} & "
        f"{r['S3DA_Acc']:.4f} & {r['min_Acc']:.4f} \\\\"
    )
lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
with open(os.path.join(OUT, "table_defect_ablation.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# ----------------------------------------------------------------------
# D) Figures
# ----------------------------------------------------------------------
print("D) figures ...")


def _plot_cloud(ax, points, title):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2,
               c=points[:, 2], cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.view_init(elev=25, azim=45)


classes = ["good", "insufficient", "excessive", "bridging", "collapse"]
gen = SolderJointGenerator(seed=42, noise=NOISE, n_points=NPOINTS)
fig, axes = plt.subplots(1, 5, figsize=(15, 3), subplot_kw=dict(projection="3d"))
for ax, c in zip(axes, classes):
    clouds, _ = gen.generate(n_per_class=1, classes=[c])
    _plot_cloud(ax, clouds[0].points, c)
fig.suptitle("Representative synthetic point clouds (one per defect class, seed 42)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sample_clouds.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# noise effect on a nominal joint
fig, axes = plt.subplots(1, 5, figsize=(15, 3), subplot_kw=dict(projection="3d"))
for i, nl in enumerate([0.005, 0.01, 0.02, 0.05, 0.10]):
    g2 = SolderJointGenerator(seed=42, noise=nl, n_points=NPOINTS)
    clouds, _ = g2.generate(n_per_class=1, classes=["good"])
    _plot_cloud(axes[i], clouds[0].points, f"good, $\\sigma$={nl}")
fig.suptitle("Sensor-noise effect on a nominal (good) joint geometry")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "noise_effect.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# defect-strength effect on insufficient
fig, axes = plt.subplots(1, 5, figsize=(15, 3), subplot_kw=dict(projection="3d"))
for i, s in enumerate([0.55, 0.65, 0.75, 0.85, 0.92]):
    g3 = SolderJointGenerator(seed=42, noise=NOISE, n_points=NPOINTS,
                              insufficient_scale_range=(s, s))
    clouds, _ = g3.generate(n_per_class=1, classes=["insufficient"])
    _plot_cloud(axes[i], clouds[0].points, f"insufficient, scale={s}")
fig.suptitle("Defect-strength effect on an insufficient-solder joint")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "defect_effect.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
summary = {
    "seeds": SEEDS,
    "multiseed": multi.to_dict(orient="records"),
    "rf_minus_gb_gap_mean": gap_mean,
    "rf_minus_gb_gap_std": gap_std,
    "noise": noise_df.to_dict(orient="records"),
    "defect": defect_df.to_dict(orient="records"),
}
with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
print("\nDONE")
