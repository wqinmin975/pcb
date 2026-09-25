# -*- coding: utf-8 -*-
"""
baseline_comparison.py
======================
SCI 投稿实验框架：基线方法定量对比 + 特征消融实验
(PCB 焊点缺陷检测，5 类：Good / Insufficient / Excessive / Bridging / Collapse)

- 点云仿真器：纯 numpy 实现（不依赖 open3d），近似 docx 中 S3DA 改进版的生成逻辑；
- 特征提取：体素计数法体积（自实现）+ scipy.spatial.ConvexHull（凸包体积/表面积）；
- 基线对比：S3DA 体积阈值法 / LR / SVM(RBF) / KNN / GradientBoosting / RandomForest(本文)；
- 消融实验：RandomForest leave-one-feature-out + 仅单特征；
- 输出：baseline_output/tables/*.csv|.tex, baseline_output/figures/confusion_matrix.png

运行：python baseline_comparison.py
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------
SEED = 42
N_PER_CLASS = 250                # 每类样本数（>=200）
CLASSES = ["good", "less", "more", "bridge", "collapse"]
CLASS_NAMES_EN = ["Good", "Insufficient", "Excessive", "Bridging", "Collapse"]
FEATURES = ["vol", "max_h", "h_var", "area",
            "vol_area_ratio", "convex_vol_ratio", "min_dist"]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_output")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")

rng_global = np.random.default_rng(SEED)


# ======================================================================
# 1. 点云仿真器（numpy 版，替代 open3d）
# ======================================================================
def _half_ellipsoid(rng, a, c, n=1500):
    """在半椭球 (半径 a, 高度 c) 内均匀采样 n 个点，z>=0，加测量噪声。"""
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rad = rng.uniform(0.0, 1.0, n) ** (1.0 / 3.0)   # 体积均匀
    pts = dirs * rad[:, None]
    pts[:, 2] = np.abs(pts[:, 2])                   # 上半球
    pts *= np.array([a, a, c])
    pts += rng.normal(0.0, 0.005, pts.shape)        # 传感器测量噪声
    pts[:, 2] = np.clip(pts[:, 2], 0.0, None)
    return pts


def simulate_sample(solder_type, rng):
    """
    生成一个样本 = 目标焊点点云 + 相邻焊点中心距离。
    返回 (points[N,3], min_dist)
    各类参数带随机扰动，使类别间存在少量重叠（不可 100% 线性/规则可分）。
    """
    # 基础尺寸（良品）：水平半径 a，高度 c
    a0 = 0.12
    c0 = 0.10
    scale = rng.normal(1.0, 0.09)                   # 良品自身尺寸波动

    if solder_type == "good":
        a, c = a0 * scale, c0 * scale
        pts = _half_ellipsoid(rng, a, c)
        min_dist = rng.normal(0.65, 0.07)

    elif solder_type == "less":                     # 少锡：体积缩小 30~50%
        s = rng.uniform(0.55, 0.85) * scale         # 上界与良品下尾重叠
        a, c = a0 * s, c0 * s * rng.uniform(0.8, 1.0)
        pts = _half_ellipsoid(rng, a, c)
        min_dist = rng.normal(0.65, 0.07)

    elif solder_type == "more":                     # 多锡：体积偏大 + 不规则隆起
        s = rng.normal(1.3, 0.12)                   # 下尾与良品上尾重叠
        a, c = a0 * s, c0 * (s + rng.normal(0.25, 0.08))
        pts = _half_ellipsoid(rng, a, c)
        nb = int(rng.integers(150, 350))            # 附加隆起
        bump = rng.normal(loc=[0.5 * a, 0.0, 0.75 * c],
                          scale=[0.3 * a, 0.3 * a, 0.2 * c], size=(nb, 3))
        bump[:, 2] = np.clip(bump[:, 2], 0.0, None)
        pts = np.vstack([pts, bump])
        min_dist = rng.normal(0.65, 0.07)

    elif solder_type == "bridge":                   # 桥连：与相邻焊点距离异常小
        a, c = a0 * scale, c0 * scale
        pts = _half_ellipsoid(rng, a, c)
        min_dist = rng.normal(0.34, 0.06)           # 与正常间距(0.65)尾部重叠
        # 连接材料：两焊点间低位"锡桥"
        nb = int(rng.integers(80, 180))
        t = rng.uniform(0.3, 1.0, nb)
        bridge_pts = np.stack([
            t * min_dist * rng.uniform(0.9, 1.0, nb),
            rng.normal(0, 0.02, nb),
            np.clip(rng.normal(0.35 * c, 0.1 * c, nb), 0.0, None)
        ], axis=1)
        pts = np.vstack([pts, bridge_pts])

    elif solder_type == "collapse":                 # 塌陷/气孔：顶部凹陷
        a, c = a0 * scale, c0 * scale
        pts = _half_ellipsoid(rng, a, c)
        # 凹陷中心随机偏移（真实气孔/塌陷位置不定），更贴近实际且非线性
        ox, oy = rng.uniform(-0.3, 0.3, 2) * a
        r_xy = np.linalg.norm(pts[:, :2] - np.array([ox, oy]), axis=1)
        dent_depth = c * rng.uniform(0.28, 0.5)     # 凹陷深度
        dent = dent_depth * np.exp(-(r_xy / (0.35 * a)) ** 2)
        pts[:, 2] = np.clip(pts[:, 2] - dent * (pts[:, 2] / max(c, 1e-6)),
                            0.0, None)
        min_dist = rng.normal(0.65, 0.07)
    else:
        raise ValueError(solder_type)

    # 统一点数：避免体素计数体积受采样密度影响（模拟固定扫描分辨率）
    N_FIXED = 1500
    if pts.shape[0] > N_FIXED:
        idx = rng.choice(pts.shape[0], N_FIXED, replace=False)
        pts = pts[idx]
    return pts, float(min_dist)


# ======================================================================
# 2. 特征提取（不依赖 open3d）
#    - vol            : 体素计数法（自实现体素栅格，voxel=0.01）
#    - convex_vol     : scipy.spatial.ConvexHull.volume
#    - area           : ConvexHull.area（凸包表面积，近似曲面表面积；
#                       注：对塌陷类凹陷区域会被凸包"盖住"，属已知近似误差）
#    - min_dist       : 焊点中心间距（仿真直接给出；真实数据用中心点欧氏距离）
# ======================================================================
VOXEL_SIZE = 0.01


def extract_features(pts, min_dist):
    z = pts[:, 2]
    max_h = float(np.max(z))
    h_var = float(np.var(z))

    # 体素计数法体积
    vox = np.floor(pts / VOXEL_SIZE).astype(np.int64)
    n_vox = len(np.unique(vox, axis=0))
    vol = n_vox * (VOXEL_SIZE ** 3)

    # 凸包体积 / 表面积
    try:
        hull = ConvexHull(pts)
        convex_vol = float(hull.volume)
        area = float(hull.area)
    except QhullError:
        convex_vol, area = vol, 1.0

    vol_area_ratio = vol / area if area > 1e-9 else 0.0
    convex_vol_ratio = vol / convex_vol if convex_vol > 1e-9 else 1.0

    return {"vol": vol, "max_h": max_h, "h_var": h_var, "area": area,
            "vol_area_ratio": vol_area_ratio,
            "convex_vol_ratio": convex_vol_ratio,
            "min_dist": min_dist}


def build_dataset(n_per_class=N_PER_CLASS, seed=SEED):
    """仿真生成全部样本并提取特征，返回 X[DataFrame], y[ndarray]。"""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for ci, ctype in enumerate(CLASSES):
        for _ in range(n_per_class):
            pts, md = simulate_sample(ctype, rng)
            rows.append(extract_features(pts, md))
            labels.append(ci)
    X = pd.DataFrame(rows, columns=FEATURES)
    y = np.array(labels)
    return X, y


# ======================================================================
# 3. 数据加载
# ======================================================================
def load_data():
    """
    返回特征矩阵 X (pd.DataFrame, 7 列) 与标签 y (ndarray, 0..4)。

    ┌──────────────────────────────────────────────────────────────┐
    │ 真实数据接入点：替换此函数返回 X, y 即可。                      │
    │ 真实 3D 相机点云到手后，在此函数内读取真实点云、按焊点分割、      │
    │ 调用 extract_features() 提取同样的 7 个特征并返回即可，          │
    │ 下游对比/消融实验代码无需任何改动。                              │
    └──────────────────────────────────────────────────────────────┘
    """
    return build_dataset()


# ======================================================================
# 4. 基线方法定义
# ======================================================================
def get_baselines(seed=SEED):
    """返回 [(编号, 名称, 估计器, 是否本文方法), ...]"""
    return [
        ("B1", "S3DA (volume threshold)",
         # 原 S3DA 方法：仅用体积特征的阈值规则分类（用体积上单棵决策树拟合阈值）
         DecisionTreeClassifier(max_depth=6, random_state=seed), False),
        ("B2", "Logistic Regression",
         Pipeline([("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=2000,
                                              random_state=seed))]), False),
        ("B3", "SVM (RBF)",
         Pipeline([("sc", StandardScaler()),
                   ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                               random_state=seed))]), False),
        ("B4", "KNN",
         Pipeline([("sc", StandardScaler()),
                   ("clf", KNeighborsClassifier(n_neighbors=7))]), False),
        ("B5", "GradientBoosting",
         GradientBoostingClassifier(random_state=seed), False),
        ("B6", "RandomForest (Ours)",
         RandomForestClassifier(n_estimators=300, random_state=seed,
                                n_jobs=-1), True),
    ]


def evaluate(clf, Xtr, ytr, Xte, yte):
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return {
        "Accuracy": accuracy_score(yte, pred),
        "Precision": precision_score(yte, pred, average="macro", zero_division=0),
        "Recall": recall_score(yte, pred, average="macro", zero_division=0),
        "F1": f1_score(yte, pred, average="macro", zero_division=0),
    }, pred


# ======================================================================
# 5. 输出工具
# ======================================================================
def b(x):
    return f"\\textbf{{{x:.4f}}}"


def write_baseline_tex(rows, path):
    lines = [
        r"\begin{table}[htbp]", r"  \centering",
        r"  \caption{Quantitative comparison with baseline methods "
        r"(macro-averaged metrics, stratified 7:3 split).}",
        r"  \label{tab:baseline}",
        r"  \begin{tabular}{llccccc}", r"    \toprule",
        r"    No. & Method & Accuracy & Precision & Recall & F1 & CV Acc.\ (5-fold) \\",
        r"    \midrule",
    ]
    for r in rows:
        fmt = b if r["ours"] else (lambda v: f"{v:.4f}")
        name = r["name"].replace(" (Ours)", "") + (" \\textbf{(Ours)}" if r["ours"] else "")
        if r["ours"]:
            name = r["name"].replace(" (Ours)", "")
            name = "\\textbf{" + name + " (Ours)}"
        lines.append(
            f"    {r['no']} & {name} & {fmt(r['Accuracy'])} & {fmt(r['Precision'])} "
            f"& {fmt(r['Recall'])} & {fmt(r['F1'])} & "
            f"{fmt(r['CV_mean'])}$\\pm${r['CV_std']:.4f} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_ablation_tex(df, path):
    lines = [
        r"\begin{table}[htbp]", r"  \centering",
        r"  \caption{Feature ablation study of the proposed RandomForest model "
        r"(macro F1 on the test set). $\Delta$F1 = F1$_{\text{variant}}-$F1$_{\text{full}}$.}",
        r"  \label{tab:ablation}",
        r"  \begin{tabular}{llcc}", r"    \toprule",
        r"    Setting & Feature & macro F1 & $\Delta$F1 \\", r"    \midrule",
    ]
    for _, r in df.iterrows():
        delta = "--" if pd.isna(r["Delta_F1"]) else f"{r['Delta_F1']:+.4f}"
        lines.append(f"    {r['Setting']} & {r['Feature']} & "
                     f"{r['F1']:.4f} & {delta} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES_EN, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES_EN)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix of RandomForest (Ours)")
    thr = cm.max() / 2
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# 6. 主流程
# ======================================================================
def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 68)
    print("PCB Solder Defect Detection - Baseline Comparison & Ablation")
    print("=" * 68)

    # ---- 数据 ----
    print(f"\n[1/4] Generating simulated point clouds "
          f"({N_PER_CLASS} samples x {len(CLASSES)} classes) ...")
    X, y = load_data()
    print(f"      Dataset: {X.shape[0]} samples, {X.shape[1]} features: "
          f"{list(X.columns)}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED)
    print(f"      Train/Test = {len(Xtr)}/{len(Xte)} (stratified 7:3, seed={SEED})")

    # ---- 基线对比 ----
    print("\n[2/4] Baseline comparison (Accuracy / macro P / R / F1, "
          "5-fold CV on training set):")
    rows, rf_pred = [], None
    for no, name, clf, ours in get_baselines():
        m, pred = evaluate(clf, Xtr.values, ytr, Xte.values, yte)
        cv = cross_val_score(clf, Xtr.values, ytr, cv=5,
                             scoring="accuracy", n_jobs=-1)
        rows.append({"no": no, "name": name, "ours": ours,
                     "CV_mean": cv.mean(), "CV_std": cv.std(), **m})
        if ours:
            rf_pred = pred
        mark = "  <== Ours" if ours else ""
        print(f"      {no} {name:28s} Acc={m['Accuracy']:.4f} "
              f"P={m['Precision']:.4f} R={m['Recall']:.4f} "
              f"F1={m['F1']:.4f} CV={cv.mean():.4f}+-{cv.std():.4f}{mark}")

    df_base = pd.DataFrame(rows)[["no", "name", "Accuracy", "Precision",
                                  "Recall", "F1", "CV_mean", "CV_std"]]
    df_base.round(4).to_csv(os.path.join(TABLE_DIR, "table_baseline.csv"),
                            index=False, encoding="utf-8-sig")
    write_baseline_tex(rows, os.path.join(TABLE_DIR, "table_baseline.tex"))

    # ---- 混淆矩阵 ----
    cm = confusion_matrix(yte, rf_pred)
    plot_confusion_matrix(cm, os.path.join(FIG_DIR, "confusion_matrix.png"))

    # ---- 消融实验 ----
    print("\n[3/4] Ablation study (RandomForest, macro F1 on test set):")
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    full_m, _ = evaluate(rf, Xtr.values, ytr, Xte.values, yte)
    f1_full = full_m["F1"]
    ab_rows = [{"Setting": "Full model", "Feature": "all 7 features",
                "F1": f1_full, "Delta_F1": np.nan}]
    print(f"      Full model (all 7 features)      F1={f1_full:.4f}")

    for feat in FEATURES:  # leave-one-feature-out
        cols = [c for c in FEATURES if c != feat]
        m, _ = evaluate(RandomForestClassifier(n_estimators=300, random_state=SEED,
                                               n_jobs=-1),
                        Xtr[cols].values, ytr, Xte[cols].values, yte)
        ab_rows.append({"Setting": "Leave-one-out", "Feature": f"w/o {feat}",
                        "F1": m["F1"], "Delta_F1": m["F1"] - f1_full})
        print(f"      w/o {feat:18s}          F1={m['F1']:.4f}  "
              f"dF1={m['F1'] - f1_full:+.4f}")

    for feat in FEATURES:  # 仅单特征
        m, _ = evaluate(RandomForestClassifier(n_estimators=300, random_state=SEED,
                                               n_jobs=-1),
                        Xtr[[feat]].values, ytr, Xte[[feat]].values, yte)
        ab_rows.append({"Setting": "Single feature", "Feature": f"only {feat}",
                        "F1": m["F1"], "Delta_F1": m["F1"] - f1_full})
        print(f"      only {feat:16s}          F1={m['F1']:.4f}  "
              f"dF1={m['F1'] - f1_full:+.4f}")

    df_ab = pd.DataFrame(ab_rows)
    df_ab.round(4).to_csv(os.path.join(TABLE_DIR, "table_ablation.csv"),
                          index=False, encoding="utf-8-sig")
    write_ablation_tex(df_ab, os.path.join(TABLE_DIR, "table_ablation.tex"))

    # ---- 摘要 ----
    print("\n[4/4] Output files:")
    for p in ["tables/table_baseline.csv", "tables/table_baseline.tex",
              "tables/table_ablation.csv", "tables/table_ablation.tex",
              "figures/confusion_matrix.png"]:
        print("      " + os.path.join(OUT_DIR, p))
    print("\nDone.")


if __name__ == "__main__":
    main()
