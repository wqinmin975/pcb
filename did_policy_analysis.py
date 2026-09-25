# -*- coding: utf-8 -*-
"""
============================================================================
DID 政策处理效应分析流水线  (Difference-in-Differences Pipeline)
============================================================================

依据文档:  Python 仿真程序.docx  (S3DA 改进版 PCB 焊点缺陷检测仿真)
----------------------------------------------------------------------------
文档是一份「合成可复现数据 + 模块化」的仿真程序（Open3D 点云 -> 几何特征 ->
RandomForest 分类 5 类焊点缺陷）。本程序沿用其写法范式，把文档中的 S3DA 改进
算法包装成一项**产线推广政策**，用标准 DID 方法估计其因果处理效应，并产出
发表级表格与事件研究图。

政策设定 (可自由改写, 见 CFG):
    Treatment group : 上线 "S3DA 改进版多特征分类器" 的产线 / 工厂
    Control group   : 仍使用传统单阈值判定法的产线 / 工厂
    Outcome         : 焊点缺陷逃逸率 (defect escape rate, %)，越低越好
    处理效应        : 上线改进算法后，缺陷逃逸率的下降幅度

流水线 (对应 empirical-research-team 的 SOP 阶段):
    Stage 0  simulate_panel()    合成面板数据（含已知处理效应，用于校验估计量）
    Stage 1  clean_data()        数据清洗：去重 / 缺失 / 面板平衡 / 缩尾 / 构造变量
    Stage 2  estimate_did()      TWFE + 2x2 + 事件研究 + 聚类稳健标准误 (虚拟变量 FE)
    Stage 3  robustness_checks() 安慰剂(置换推断) / 双向聚类 / 平衡样本 / 敏感性
    Stage 4  build_tables()      发表级表格 (booktabs .tex + .csv) 与事件研究图

实现要点 (保证发表级严谨):
    * 固定效应用「实体 + 时间虚拟变量」吸收 (对非平衡面板等价于 FE, 已与 statsmodels 交叉验证一致)
    * 聚类稳健标准误用 CRVE, 并报告单向(按产线)与双向(产线 x 时期, Cameron-Gelbach-Miller)两种
依赖:  numpy, pandas, matplotlib  (无需 linearmodels)
用法:  python did_policy_analysis.py
============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import erf, sqrt

# ----------------------------- 全局配置 -----------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "did_output")
TBL_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(TBL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CFG = dict(
    N=150, treat_frac=0.5, T=12, adopt_period=6,
    seed=20240923, true_te=-1.20, baseline=5.0, sigma_noise=0.8,
    miss_frac=0.03, n_dup=2, win_pct=(0.01, 0.99), perm_trials=500,
)


# ===========================================================================
# 计量工具: 虚拟变量固定效应 OLS + 聚类稳健标准误 (CRVE / CGM 双向)
# ===========================================================================
def _meat(X, e, groups):
    """单路聚类 meat 矩阵 Σ_g (X_g'e_g)(X_g'e_g)' 与组数 G。"""
    groups = np.asarray(groups)
    k = X.shape[1]
    meat = np.zeros((k, k))
    for g in np.unique(groups):
        m = groups == g
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    return meat, int(np.unique(groups).size)


def _pairkey(a, b):
    return np.array([f"{x}-{y}" for x, y in zip(np.asarray(a), np.asarray(b))], dtype=object)


def _crve_se(X, e, groups):
    n, k = X.shape
    meat, G = _meat(X, e, groups)
    bread = np.linalg.pinv(X.T @ X)
    corr = (G / (G - 1)) * ((n - 1) / (n - k))
    V = bread @ meat @ bread * corr
    return np.sqrt(np.clip(np.diag(V), 0, None))


def fit_ols(X, y, cluster1, cluster2=None):
    """OLS + 单向/双向(CGM)聚类稳健标准误。返回 (beta, se_oneway, se_twoway)。"""
    beta = np.linalg.pinv(X) @ y
    e = y - X @ beta
    se1 = _crve_se(X, e, cluster1)
    if cluster2 is not None:
        n, k = X.shape
        m1, G1 = _meat(X, e, cluster1)
        m2, G2 = _meat(X, e, cluster2)
        m12, G12 = _meat(X, e, _pairkey(cluster1, cluster2))
        bread = np.linalg.pinv(X.T @ X)

        def V(m, G):
            return bread @ m @ bread * ((G / (G - 1)) * ((n - 1) / (n - k)))
        se2 = np.sqrt(np.clip(np.diag(V(m1, G1) + V(m2, G2) - V(m12, G12)), 0, None))
    else:
        se2 = se1
    return beta, se1, se2


def build_spec(df, entity=False, time=False, treat_main=False,
               did_arr=None, covs=None, extras=None):
    """构造设计矩阵, 返回 (X, did_idx, cov_idxs, extra_idxs)。"""
    cols = [np.ones(len(df))]
    if entity:
        for i in sorted(df["id"].unique())[1:]:
            cols.append((df["id"] == i).astype(float))
    if time:
        for t in sorted(df["time"].unique())[1:]:
            cols.append((df["time"] == t).astype(float))
    if treat_main:
        cols.append(df["treated"].values.astype(float))
    extra_idxs = []
    if extras:
        for ex in extras:
            cols.append(np.asarray(ex, float))
            extra_idxs.append(len(cols) - 1)
    did_idx = None
    if did_arr is not None:
        cols.append(np.asarray(did_arr, float))
        did_idx = len(cols) - 1
    cov_idxs = []
    if covs:
        for c in covs:
            cols.append(df[c].values.astype(float))
            cov_idxs.append(len(cols) - 1)
    return np.column_stack(cols), did_idx, cov_idxs, extra_idxs


# ===========================================================================
# Stage 0  合成面板数据
# ===========================================================================
def simulate_panel(cfg=None):
    cfg = cfg or CFG
    rng = np.random.default_rng(cfg["seed"])
    N, T, adopt = cfg["N"], cfg["T"], cfg["adopt_period"]

    ids = np.repeat(np.arange(N), T)
    times = np.tile(np.arange(T), N)
    unit_fe = rng.normal(0.0, 1.0, size=N)[ids]
    time_fe = (0.05 * times) + rng.normal(0.0, 0.15, size=N * T)
    size = rng.normal(0.0, 1.0, size=N)[ids]
    throughput = rng.normal(0.0, 1.0, size=N * T)

    treated = (rng.uniform(size=N) < cfg["treat_frac"]).astype(int)[ids]
    post = (times >= adopt).astype(int)
    did = treated * post

    y = (cfg["baseline"] + unit_fe + time_fe + cfg["true_te"] * did
         + 0.15 * size + 0.10 * throughput
         + rng.normal(0.0, cfg["sigma_noise"], size=N * T))

    df = pd.DataFrame({
        "id": ids, "time": times, "outcome": y,
        "treated": treated, "post": post, "did": did,
        "size": size, "throughput": throughput,
    })

    # 注入真实数据常见瑕疵, 演示 Stage 1 清洗
    drop_idx = rng.choice(df.index, size=int(cfg["miss_frac"] * len(df)), replace=False)
    df = df.drop(index=drop_idx).reset_index(drop=True)
    dup_rows = df.sample(n=cfg["n_dup"], random_state=cfg["seed"]).copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    out_idx = rng.choice(df.index, size=5, replace=False)
    df.loc[out_idx, "outcome"] += rng.choice([12.0, -10.0], size=5)

    df["rel_time"] = df["time"] - adopt
    return df


# ===========================================================================
# Stage 1  数据清洗
# ===========================================================================
def clean_data(df, cfg=None):
    cfg = cfg or CFG
    diag = {}
    n_raw = len(df)
    df = df.drop_duplicates(subset=["id", "time"]).reset_index(drop=True)
    diag["rows_raw"] = n_raw
    diag["rows_dup_removed"] = n_raw - len(df)
    diag["missing_cells"] = int(df.isna().sum().sum())

    cnt = df.groupby("id").size()
    diag["n_units"] = int(df["id"].nunique())
    diag["n_periods"] = int(df["time"].nunique())
    diag["balanced_units"] = int((cnt == cfg["T"]).sum())
    diag["unbalanced_units"] = diag["n_units"] - diag["balanced_units"]

    df["outcome_unw"] = df["outcome"].copy()      # 保留未缩尾版供稳健性敏感性
    lo, hi = cfg["win_pct"]

    def winsor(s):
        ql, qh = s.quantile(lo), s.quantile(hi)
        return s.clip(ql, qh)

    for c in ["outcome", "size", "throughput"]:
        df[c] = winsor(df[c])
    diag["rows_clean"] = len(df)
    return df, diag


# ===========================================================================
# Stage 2  估计: 2x2 / TWFE(M1-M5) / 事件研究
# ===========================================================================
def estimate_did(df, cfg=None):
    cfg = cfg or CFG
    idv, timev = df["id"].values, df["time"].values
    y = df["outcome"].values
    did = df["did"].values.astype(float)
    res = {}

    # M1: 无 FE 的 2x2
    X1, di1, _, _ = build_spec(df, did_arr=did,
                               extras=[df["treated"].values.astype(float),
                                       df["post"].values.astype(float)])
    # 上面 extras 顺序: [treated, post]; did 在最后 -> did_idx 已知
    b1, s1, _ = fit_ols(X1, y, idv)
    res["M1"] = (float(b1[di1]), float(s1[di1]))

    # M2: 仅单位 FE
    X2, di2, _, _ = build_spec(df, entity=True, did_arr=did)
    b2, s2, _ = fit_ols(X2, y, idv)
    res["M2"] = (float(b2[di2]), float(s2[di2]))

    # M3: 仅时间 FE (含 treated 主效应)
    X3, di3, _, _ = build_spec(df, time=True, treat_main=True, did_arr=did)
    b3, s3, _ = fit_ols(X3, y, idv)
    res["M3"] = (float(b3[di3]), float(s3[di3]))

    # M4: TWFE (单位 + 时间 FE)
    X4, di4, _, _ = build_spec(df, entity=True, time=True, did_arr=did)
    b4, s4_1, s4_2 = fit_ols(X4, y, idv, timev)
    res["M4"] = (float(b4[di4]), float(s4_1[di4]), float(s4_2[di4]))

    # M5: TWFE + 时间变化协变量 (注意: 时间不变协变量会被实体 FE 吸收, 不能放入)
    X5, di5, ci5, _ = build_spec(df, entity=True, time=True, did_arr=did,
                                 covs=["throughput"])
    b5, s5_1, s5_2 = fit_ols(X5, y, idv, timev)
    res["M5"] = (float(b5[di5]), float(s5_1[di5]), float(s5_2[di5]))
    res["M5_thru"] = (float(b5[ci5[0]]), float(s5_1[ci5[0]]))
    res["N"] = int(len(y))

    # 事件研究: 相对事件时间虚拟 (基期 rel_time = -1), 仅对处理单位触发。
    # 注意: 单时点采用下 rel_time 是 time 的确定性函数, 若对全样本设虚拟会与时间 FE 完全共线;
    #         正确做法是 D^k_it = 1 仅当 (treated_i==1 且 rel_time_it==k)。
    rel = df["rel_time"].values
    ks = [int(k) for k in np.unique(rel) if k != -1]
    treated_flag = df["treated"].values == 1
    extras = [((treated_flag) & (rel == k)).astype(float) for k in ks]
    Xev, _, _, ei = build_spec(df, entity=True, time=True, extras=extras)
    bev, sev, _ = fit_ols(Xev, y, idv)
    coef = [float(bev[j]) for j in ei]
    se_ev = [float(sev[j]) for j in ei]
    res["event"] = {
        "ks": ks,
        "coef": coef,
        "lo": [c - 1.96 * s for c, s in zip(coef, se_ev)],
        "hi": [c + 1.96 * s for c, s in zip(coef, se_ev)],
    }
    return res


# ===========================================================================
# Stage 3  稳健性检验
# ===========================================================================
def _twfe(df, y, did_arr, cluster2=True):
    X, di, _, _ = build_spec(df, entity=True, time=True, did_arr=did_arr)
    b, s1, s2 = fit_ols(X, y, df["id"].values, df["time"].values if cluster2 else None)
    return float(b[di]), float(s1[di]), float(s2[di])


def robustness_checks(df, main_coef, cfg=None):
    cfg = cfg or CFG
    idv = df["id"].values
    y = df["outcome"].values
    did = df["did"].values.astype(float)
    rows = []

    # R1 安慰剂: 随机置换处理身份 (保持采用时点)
    rng = np.random.default_rng(cfg["seed"] + 1)
    treat = df["treated"].values.astype(int).copy()
    perm = []
    for _ in range(cfg["perm_trials"]):
        shuffled = rng.permutation(treat)
        didp = (shuffled * df["post"].values).astype(float)
        bp, _, _ = _twfe(df, y, didp, cluster2=False)
        perm.append(bp)
    perm = np.array(perm)
    pval = float(np.mean(np.abs(perm) >= abs(main_coef)))
    rows.append(("安慰剂(置换推断) 系数均值", float(perm.mean()), float(perm.std()), "ref"))
    rows.append(("安慰剂(置换推断) p 值", float(main_coef), float(perm.std()), f"p={pval:.3f}"))
    perm_p = pval

    # R2 双向聚类 vs 单向聚类 SE
    _, s4_1, s4_2 = _twfe(df, y, did, cluster2=True)
    rows.append(("双向聚类 SE (TWFE)", float(s4_2), float(s4_1), "对比单向"))

    # R3 仅平衡面板
    cnt = df.groupby("id").size()
    bal_ids = cnt[cnt == cfg["T"]].index
    dfb = df[df["id"].isin(bal_ids)]
    bb, sb, _ = _twfe(dfb, dfb["outcome"].values, dfb["did"].values.astype(float), cluster2=False)
    rows.append((f"仅平衡面板 (N={len(bal_ids)})", float(bb), float(sb), "TWFE"))

    # R4 剔除极值单位 (按采用前均值前/后 5%)
    pre = df[df["post"] == 0].groupby("id")["outcome"].mean()
    lo_q, hi_q = pre.quantile(0.05), pre.quantile(0.95)
    keep = pre[(pre >= lo_q) & (pre <= hi_q)].index
    dfr = df[df["id"].isin(keep)]
    br, sr, _ = _twfe(dfr, dfr["outcome"].values, dfr["did"].values.astype(float), cluster2=False)
    rows.append(("剔除前5%/后5%单位", float(br), float(sr), "TWFE"))

    # R5 缩尾敏感性: 用未缩尾 outcome 重算 TWFE
    bu, su, _ = _twfe(df, df["outcome_unw"].values, did, cluster2=False)
    rows.append(("未缩尾敏感性 (TWFE)", float(bu), float(su), "vs 主表缩尾"))

    return rows, perm_p


# ===========================================================================
# Stage 4  发表级表格 + 图形
# ===========================================================================
def _z_to_p(z):
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def stars(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))


def _cell(val, se):
    if val is None:
        return "—"
    p = _z_to_p(val / se) if (se and se > 0) else 1.0
    return f"{val:.3f}{stars(p)} \\\\ ({se:.3f})"


def table1_descriptive(df):
    cols = ["outcome", "size", "throughput"]
    out = []
    for c in cols:
        tg = df[df["treated"] == 1][c]
        cg = df[df["treated"] == 0][c]
        nx, ny = len(tg), len(cg)
        vx, vy = tg.var(), cg.var()
        t = (tg.mean() - cg.mean()) / np.sqrt(vx / nx + vy / ny)
        p = _z_to_p(t)
        out.append((c, tg.mean(), tg.std(), cg.mean(), cg.std(), tg.mean() - cg.mean(), p))
    return out


def save_table1(df):
    rows = table1_descriptive(df)
    csv = "variable,treat_mean,treat_sd,control_mean,control_sd,diff,pvalue\n"
    for r in rows:
        csv += f"{r[0]},{r[1]:.3f},{r[2]:.3f},{r[3]:.3f},{r[4]:.3f},{r[5]:.3f},{r[6]:.3f}\n"
    p_csv = os.path.join(TBL_DIR, "table1_descriptive.csv")
    with open(p_csv, "w", encoding="utf-8") as f:
        f.write(csv)
    tex = ["\\begin{table}[htbp]\\centering\\caption{描述统计: 处理组 vs 控制组}",
           "\\label{tab:desc}", "\\begin{tabular}{lcccccc}\\toprule",
           "变量 & 处理组均值 & 处理组SD & 控制组均值 & 控制组SD & 差异 & p值 \\\\ \\midrule"]
    for r in rows:
        tex.append(f"{r[0]} & {r[1]:.3f} & {r[2]:.3f} & {r[3]:.3f} & {r[4]:.3f} "
                   f"& {r[5]:.3f} & {r[6]:.3f} \\\\")
    tex += ["\\bottomrule\\end{tabular}\\end{table}"]
    p_tex = os.path.join(TBL_DIR, "table1_descriptive.tex")
    with open(p_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    return p_csv, p_tex


def save_table2(res):
    c1, s1 = res["M1"]; c2, s2 = res["M2"]; c3, s3 = res["M3"]
    c4, s4_1, s4_2 = res["M4"]; c5, s5_1, s5_2 = res["M5"]
    ct, st = res["M5_thru"]

    tex = ["\\begin{table}[htbp]\\centering",
           "\\caption{政策处理效应: DID 估计 (聚类稳健标准误, 按产线聚类)}",
           "\\label{tab:did}", "\\begin{tabular}{lccccc}\\toprule",
           " & (1) & (2) & (3) & (4) & (5) \\\\",
           " & 无FE & 单位FE & 时间FE & TWFE & TWFE+协变量 \\\\ \\midrule"]
    tex.append(f"did (处理$\\times$后) & {_cell(c1,s1)} & {_cell(c2,s2)} & {_cell(c3,s3)} "
               f"& {_cell(c4,s4_1)} & {_cell(c5,s5_1)} \\\\")
    tex.append(f"throughput (时变控制) & — & — & — & — & {_cell(ct,st)} \\\\")
    tex.append("\\midrule")
    tex.append(f"N & {res['N']} & {res['N']} & {res['N']} & {res['N']} & {res['N']} \\\\")
    tex.append("单位 FE & 否 & 是 & 否 & 是 & 是 \\\\")
    tex.append("时间 FE & 否 & 否 & 是 & 是 & 是 \\\\")
    tex.append("聚类 & 产线 & 产线 & 产线 & 产线 & 产线 \\\\")
    tex.append("\\bottomrule\\end{tabular}\\end{table}")
    p_tex = os.path.join(TBL_DIR, "table2_did.tex")
    with open(p_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))

    csv = "spec,coef,se_oneway,se_twoway\n"
    csv += f"M1_noFE,{c1:.4f},{s1:.4f},\n"
    csv += f"M2_entityFE,{c2:.4f},{s2:.4f},\n"
    csv += f"M3_timeFE,{c3:.4f},{s3:.4f},\n"
    csv += f"M4_TWFE,{c4:.4f},{s4_1:.4f},{s4_2:.4f}\n"
    csv += f"M5_TWFE_cov,{c5:.4f},{s5_1:.4f},{s5_2:.4f}\n"
    p_csv = os.path.join(TBL_DIR, "table2_did.csv")
    with open(p_csv, "w", encoding="utf-8") as f:
        f.write(csv)
    return p_csv, p_tex


def save_table_event(res):
    ev = res["event"]
    csv = "rel_time,coef,ci_lo,ci_hi\n"
    for k, c, lo, hi in zip(ev["ks"], ev["coef"], ev["lo"], ev["hi"]):
        csv += f"{k},{c:.4f},{lo:.4f},{hi:.4f}\n"
    p_csv = os.path.join(TBL_DIR, "table3_event_study.csv")
    with open(p_csv, "w", encoding="utf-8") as f:
        f.write(csv)

    ks = np.array(ev["ks"]); co = np.array(ev["coef"])
    lo = np.array(ev["lo"]); hi = np.array(ev["hi"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(ks, co, yerr=[co - lo, hi - co], fmt="o-", color="#1f77b4",
                ecolor="#999999", capsize=3, label="Event-study coefficient")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(-0.5, color="red", lw=0.8, ls=":")
    ax.set_xlabel("Relative event time (rel_time, base = -1)")
    ax.set_ylabel("Change in defect escape rate (pp)")
    ax.set_title("Event study: parallel trends & dynamic effects")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p_png = os.path.join(FIG_DIR, "event_study.png")
    fig.savefig(p_png, dpi=300); plt.close(fig)
    return p_csv, p_png


def save_table_robust(rows):
    tex = ["\\begin{table}[htbp]\\centering\\caption{稳健性检验}",
           "\\label{tab:rob}", "\\begin{tabular}{lccc}\\toprule",
           "检验 & 系数/指标 & 标准误/离散 & 备注 \\\\ \\midrule"]
    for name, v, se, note in rows:
        tex.append(f"{name} & {v:.3f} & {se:.3f} & {note} \\\\")
    tex.append("\\bottomrule\\end{tabular}\\end{table}")
    p_tex = os.path.join(TBL_DIR, "table4_robustness.tex")
    with open(p_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    csv = "check,value,se,note\n"
    for name, v, se, note in rows:
        csv += f"{name},{v:.4f},{se:.4f},{note}\n"
    p_csv = os.path.join(TBL_DIR, "table4_robustness.csv")
    with open(p_csv, "w", encoding="utf-8") as f:
        f.write(csv)
    return p_csv, p_tex


# ===========================================================================
# 主流程
# ===========================================================================
def main():
    print("=" * 70)
    print("DID 政策处理效应分析流水线")
    print("=" * 70)

    df_raw = simulate_panel(CFG)
    print(f"[Stage0] 合成面板: {len(df_raw)} 行 (含注入瑕疵), 真实处理效应 = {CFG['true_te']}")

    df, diag = clean_data(df_raw, CFG)
    print(f"[Stage1] 清洗: 去重 {diag['rows_dup_removed']} 行, 缺失单元格 "
          f"{diag['missing_cells']}, 不平衡单位 {diag['unbalanced_units']}, 清洗后 {diag['rows_clean']} 行")

    res = estimate_did(df, CFG)
    print(f"[Stage2] M1(2x2)={res['M1'][0]:.3f}  M2(单位FE)={res['M2'][0]:.3f}  "
          f"M3(时间FE)={res['M3'][0]:.3f}")
    print(f"[Stage2] M4(TWFE)={res['M4'][0]:.3f} (se 单向 {res['M4'][1]:.3f} / 双向 {res['M4'][2]:.3f})")
    print(f"[Stage2] M5(TWFE+协)={res['M5'][0]:.3f} (se {res['M5'][1]:.3f})")

    rob, perm_p = robustness_checks(df, res["M4"][0], CFG)
    for r in rob:
        print(f"[Stage3] {r[0]}: {r[1]:.3f} ({r[3]})")

    t1 = save_table1(df)
    t2 = save_table2(res)
    t3 = save_table_event(res)
    t4 = save_table_robust(rob)
    print(f"[Stage4] 表格/图形已写出:\n  {t1}\n  {t2}\n  {t3}\n  {t4}")

    seal = {
        "true_te": CFG["true_te"], "est_M1_2x2": res["M1"][0], "est_M4_TWFE": res["M4"][0],
        "est_M5_TWFE_cov": res["M5"][0], "se_TWFE_oneway": res["M4"][1],
        "se_TWFE_twoway": res["M4"][2], "N": res["N"], "T": CFG["T"],
        "adopt_period": CFG["adopt_period"], "seed": CFG["seed"], "perm_pvalue": perm_p,
    }
    with open(os.path.join(OUT_DIR, "result.json"), "w", encoding="utf-8") as f:
        json.dump(seal, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"完成. 输出目录: {OUT_DIR}")
    print(f"校验: 估计 TWFE={res['M4'][0]:.3f} 应逼近真实处理效应 {CFG['true_te']}")
    print("=" * 70)
    return df, res, rob


if __name__ == "__main__":
    main()
