#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seasonal_consistency_rank.py
============================

用途
----
读取“季节一致性”统计结果，计算综合相似度分数 S，并输出：
1) 每个区域的横向条形图（标注 Top-5） → figures/seasonal_rank_regionwise_<region>.png
2) 全局 Top-K（默认10）横向条形图 → figures/seasonal_rank_global.png
3) 汇总 CSV（各区域 S 与全局均分） → output/seasonal_rank_summary.csv

分数定义
--------
S = (1/3)*r + (1/3)*(1 - |amp_ratio - 1|) + (1/3)*(1 - |dphi|/180)

输入
----
默认从 --indir 目录读取所有 CSV（推荐传入 output/tables）。
脚本自动兼容下列列名（不区分大小写）：
- r:            直接列名 r，或 shape_r
- amp_ratio:    直接列名 amp_ratio，或 A_ratio；若都没有但有 std_ratio，则以 std_ratio 近似 amp_ratio
- dphi:         直接列名 dphi，或 dphi_deg，或 delta_phase_deg
- region:       若表中没有 region 列，则从文件名推断（例如 SEASONAL_consistency_..._NEP.csv → NEP）

快速使用
--------
python scripts/seasonal_consistency_rank.py --indir output/tables
python scripts/seasonal_consistency_rank.py --indir output/tables --topk 15

参数
----
--indir       输入目录（包含各区 CSV），默认 output/tables
--outdir_fig  图输出目录（默认 output/figures）
--outdir_tab  表输出目录（默认 output）
--topk        全局 TopK 模型数量（默认 10）

依赖
----
pandas, numpy, matplotlib
"""

from __future__ import annotations
from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe 


# ---------------------------- I/O & Utils ----------------------------

def log(msg: str) -> None:
    """简单的日志打印。"""
    print(f"[seasonal_rank] {msg}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seasonal consistency ranking (region-wise & global).")
    p.add_argument("--indir",  default="output/tables", help="输入目录（包含各区 CSV）。")
    p.add_argument("--outdir_fig", default="output/figures", help="图输出目录。")
    p.add_argument("--outdir_tab", default="output", help="表输出目录。")
    p.add_argument("--topk", type=int, default=10, help="全局TopK显示数量（默认10）。")
    p.add_argument("--agg", choices=["mean", "max", "median"], default="mean",
               help="同一区域同一模型有多行时如何聚合S：mean/max/median，默认 mean")
    return p.parse_args()

def ensure_dirs(*paths: str | Path) -> None:
    """确保目录存在。"""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def wrap_dphi(deg: float) -> float:
    """将相位差规范化到 [-180, 180)。"""
    return (float(deg) + 180.0) % 360.0 - 180.0

def infer_region_from_filename(name: str) -> str | None:
    """
    尝试从文件名中推断 region（NEP/NEA/SEP/SEA/SEI）。
    例如：SEASONAL_consistency_cllisccp_NEA.csv -> NEA
    """
    m = re.search(r"_(NEP|NEA|SEP|SEA|SEI)\.csv$", name, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


# ---------------------------- Core Logic ----------------------------

def load_all_csv(indir: Path) -> pd.DataFolder:
    files = sorted(indir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"未找到任何 CSV：{indir}")

    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        cols = {c.lower(): c for c in df.columns}

        # 模型名
        mcol = None
        for k in ["model", "mod", "model_id", "name"]:
            if k in cols: mcol = cols[k]; break
        if mcol is None:
            continue

        # 计算/推断 region
        rcol = cols.get("region")
        if rcol is not None:
            region_series = df[rcol].astype(str)
        else:
            # 从文件名末尾推断 …_<REG>.csv
            stem = f.stem
            reg = None
            for tag in ["NEP","NEA","SEP","SEA","SEI"]:
                if stem.upper().endswith("_"+tag) or f.name.upper().endswith("_"+tag+".CSV"):
                    reg = tag; break
            if reg is None:
                continue
            region_series = pd.Series([reg]*len(df))

        # 形状相关 r
        rcol_guess = None
        for k in ["r","shape_r","corr","corr12","corr_clim12","rm","rm_pearson"]:
            if k in cols: rcol_guess = cols[k]; break
        r_val = pd.to_numeric(df[rcol_guess], errors="coerce") if rcol_guess else np.nan

        # 幅度比：优先 amp_ratio；否则用 A_model/A_obs
        if "amp_ratio" in cols:
            amp = pd.to_numeric(df[cols["amp_ratio"]], errors="coerce")
        else:
            am = cols.get("a_model") or cols.get("amodel") or cols.get("amod") or cols.get("a_model_1")
            ao = cols.get("a_obs")   or cols.get("aobs")   or cols.get("a_ref")  or cols.get("a_ref1")
            if am and ao:
                amp = pd.to_numeric(df[am], errors="coerce") / pd.to_numeric(df[ao], errors="coerce")
            else:
                amp = pd.Series(np.nan, index=df.index)

        # 相位差 Δφ：优先 dphi / delta_phase_deg；否则 φ_model - φ_obs
        if "dphi" in cols:
            dphi = pd.to_numeric(df[cols["dphi"]], errors="coerce")
        elif "delta_phase_deg" in cols:
            dphi = pd.to_numeric(df[cols["delta_phase_deg"]], errors="coerce")
        else:
            pm = cols.get("phi_model_deg") or cols.get("phi_model") or cols.get("phi_m")
            po = cols.get("phi_obs_deg")   or cols.get("phi_obs")   or cols.get("phi_o")
            if pm and po:
                dphi = pd.to_numeric(df[pm], errors="coerce") - pd.to_numeric(df[po], errors="coerce")
            else:
                dphi = pd.Series(np.nan, index=df.index)

        part = pd.DataFrame({
            "model": df[mcol].astype(str).str.strip(),
            "region": region_series.str.upper().str.strip(),
            "r": r_val,
            "amp_ratio": amp,
            "dphi": dphi
        })
        rows.append(part)
    if not rows:
        raise RuntimeError(f"目录下 CSV 解析不到需要的字段：{indir}")

    out = pd.concat(rows, ignore_index=True)
    # 规范 Δφ 到 [-180,180]
    out["dphi"] = ((pd.to_numeric(out["dphi"], errors="coerce") + 180) % 360) - 180
    return out



def compute_S(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算综合相似度分数 S：
        S = (1/3)*r + (1/3)*(1 - |amp_ratio - 1|) + (1/3)*(1 - |dphi|/180)
    返回添加 S 列的 DataFrame（保留原始列）。
    """
    df = df.copy()
    r = np.clip(df["r"].astype(float), -1.0, 1.0)
    amp_ratio = df["amp_ratio"].astype(float)
    dphi = df["dphi"].astype(float)
    term_r = r
    term_amp = 1.0 - np.abs(amp_ratio - 1.0)
    term_phase = 1.0 - (np.abs(dphi) / 180.0)
    df["S"] = (term_r + term_amp + term_phase) / 3.0
    return df


# ---------------------------- Plotting ----------------------------

REGION_COLORS = {
    "NEP": "#1f77b4",  # 蓝
    "NEA": "#ff7f0e",  # 橙
    "SEP": "#2ca02c",  # 绿
    "SEA": "#d62728",  # 红
    "SEI": "#9467bd",  # 紫
}

def plot_regionwise(df_reg: pd.DataFrame, region: str, out_png: Path) -> None:
    """
    画某区域的横向条形图（所有模型），并标注前5名。
    """
    if df_reg.empty:
        log(f"[{region}] 无数据，跳过画图。")
        return
    d = df_reg.sort_values("S", ascending=True)  # barh 从下到上
    models = d["model"].astype(str).values
    scores = d["S"].values

    # 根据条目数调整图高
    h = max(3.0, 0.28 * len(d) + 1.4)
    plt.figure(figsize=(10, h))
    color = REGION_COLORS.get(region, "#555555")
    bars = plt.barh(models, scores, color=color, alpha=0.85)
    plt.xlabel("Similarity score S", fontsize=12)
    plt.title(f"Seasonal consistency — {region} (region-wise ranking)", fontsize=13)
    plt.grid(axis="x", alpha=0.25)

    # 只给 Top-5 标注，避免拥挤
    d_top = df_reg.sort_values("S", ascending=False).head(5).reset_index(drop=True)
    top_set = set(d_top["model"].astype(str).values)
    
    # 建议统一风格：柱内标白描边，柱外标黑色
    for rect, m, s in zip(bars, models, scores):
        if m not in top_set:
            continue
        # 判定是否“柱内标注”
        inside = s >= 0.85  # 分数很高时，放柱内更美观
        if inside:
            x_txt = s - 0.015      # 往左回一点
            ha    = "right"
            color = "white"
            kw_effects = [pe.withStroke(linewidth=2.2, foreground="black", alpha=0.6)]
        else:
            x_txt = s + 0.012      # 柱外右侧
            ha    = "left"
            color = "black"
            kw_effects = None
    
        plt.text(
            x_txt,
            rect.get_y() + rect.get_height()/2,
            f"{s:.3f}",
            va="center",
            ha=ha,
            fontsize=10.5,
            fontweight="bold",
            color=color,
            path_effects=kw_effects
        )
    
    # 轴与网格的细节（发表级）
    ax = plt.gca()
    ax.set_xlim(0.0, 1.0)            # S 的定义落在 [0,1]，固定轴限更统一
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.grid(axis="x", alpha=0.25, linestyle=":", linewidth=0.8)
    
    # 更高分辨率保存
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_global_top(df: pd.DataFrame, topk: int, out_png: Path) -> None:
    """
    画全局TopK的横向条形图（平均 S），y 轴模型名，x 为平均 S。
    """
    if df.empty:
        log("全局无数据，跳过全局图。")
        return
    g = df.groupby("model", as_index=False)["S"].mean().rename(columns={"S": "S_global"})
    g = g.sort_values("S_global", ascending=False).head(topk)
    models = g["model"].astype(str).values[::-1]  # 反转让最高在最上
    scores = g["S_global"].values[::-1]

    plt.figure(figsize=(10, max(3.0, 0.5*len(g)+1.5)))
    # 更亮的颜色
    bars = plt.barh(models, scores, color="#00bcd4", alpha=0.9)
    plt.xlabel("Global mean S across regions", fontsize=12)
    plt.title(f"Global Top {topk} seasonal consistency (mean S)", fontsize=13)
    plt.grid(axis="x", alpha=0.25)

    # 在条右侧标注
    for rect, s in zip(bars, scores):
        inside = s >= 0.85
        if inside:
            x_txt = s - 0.015; ha = "right"; color = "white"
            effects = [pe.withStroke(linewidth=2.2, foreground="black", alpha=0.6)]
        else:
            x_txt = s + 0.012; ha = "left";  color = "black"
            effects = None
        plt.text(
            x_txt, rect.get_y() + rect.get_height()/2, f"{s:.3f}",
            va="center", ha=ha, fontsize=10.5, fontweight="bold",
            color=color, path_effects=effects
        )
    
    ax = plt.gca()
    ax.set_xlim(0, 1); ax.set_xticks(np.linspace(0, 1, 6))
    ax.tick_params(axis="x", labelsize=10); ax.tick_params(axis="y", labelsize=10)
    for spine in ["top","right"]: ax.spines[spine].set_visible(False)
    plt.grid(axis="x", alpha=0.25, linestyle=":", linewidth=0.8)
    plt.tight_layout(); plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close()



# ---------------------------- Main Flow ----------------------------
'''
def main():
    args = parse_args()
    indir = Path(args.indir)
    out_fig = Path(args.outdir_fig)
    out_tab = Path(args.outdir_tab)
    ensure_dirs(out_fig, out_tab)

    log(f"读取目录：{indir}")
    df = load_all_csv(indir)

    log("计算综合分数 S")
    df = compute_S(df)

    # 统一 region 字段
    df["region"] = df["region"].astype(str).str.upper()
    agg_fun = {"mean": "mean", "max": "max", "median": "median"}[args.agg]
    df_agg = (df
              .groupby(["region","model"], as_index=False)
              .agg(S=( "S", agg_fun )) )

    # 区域内排名 & 绘图
    for region, df_reg in df.groupby("region"):
        df_reg = df_reg.copy()
        df_reg = df_reg.sort_values("S", ascending=False)
        fig_path = out_fig / f"seasonal_rank_regionwise_{region}.png"
        plot_regionwise(df_reg, region, fig_path)

    # 全局 TopK
    fig_global = out_fig / "seasonal_rank_global.png"
    plot_global_top(df, args.topk, fig_global)

    # 汇总表（每模型各区S与全局均分）
    log("输出汇总表 CSV")
    pivot = df.pivot_table(index="model", columns="region", values="S", aggfunc="mean")
    global_mean = df.groupby("model")["S"].mean().rename("S_global")
    summary = pivot.join(global_mean, how="left").reset_index()
    csv_path = out_tab / "seasonal_rank_summary.csv"
    summary.to_csv(csv_path, index=False)
    log(f"完成：{csv_path}")
'''
def main():
    # 1) 解析参数 & 目录
    args = parse_args()
    indir = Path(getattr(args, "indir", "output/tables"))
    fig_dir = Path(getattr(args, "outdir_fig", "output/figures"))
    tbl_dir = Path(getattr(args, "outdir", "output"))
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    log(f"读取目录：{indir}")
    df_raw = load_all_csv(indir)
    log(f"读取完成，共 {len(df_raw)} 行")

    # 2) 计算 S 指标
    log("计算综合得分 S (基于 r / amp_ratio / dphi)")
    df_scored = compute_S(df_raw)

    # 3) 同一区域同一模型的多行记录去重/聚合（防止绘图重叠）
    agg_key = getattr(args, "agg", "mean")
    agg_map = {"mean": "mean", "max": "max", "median": "median"}
    agg_fun = agg_map.get(agg_key, "mean")
    if agg_key not in agg_map:
        log(f"未识别的聚合方式 '{agg_key}'，已回退为 'mean'")
    log(f"按 region×model 以 {agg_fun} 方式聚合 S")
    df_agg = (
        df_scored.groupby(["region", "model"], as_index=False)
                 .agg(S=("S", agg_fun))
    )
    if df_agg.empty:
        raise RuntimeError("聚合后无可用数据，请检查输入列名或表结构。")

    # 4) 区域榜单绘图（仅对每区 Top-5 标注数值）
    for region in sorted(df_agg["region"].dropna().unique()):
        reg_df = df_agg.loc[df_agg["region"] == region].copy()
        reg_df.sort_values("S", ascending=False, inplace=True)
        out_png = fig_dir / f"seasonal_rank_{region}.png"   # 统一命名；便于投稿整理
        plot_result = plot_regionwise(reg_df, region, out_png)  # 函数内已做高分辨率与智能标注
        log(f"区域 {region}: {len(reg_df)} 个模型，已生成 {out_png}")

    # 5) 全局 Top-N（默认 Top-10）图
    log("绘制全局 Top-N 图")
    # 直接基于聚合后的 df_agg 计算跨区均值
    topN = getattr(args, "top", getattr(args, "top_n", getattr(args, "topn", getattr(args, "topN", None))))
    topN = topN if isinstance(topN, int) else getattr(args, "top", None)
    topN = getattr(args, "top", None) or getattr(args, "top_n", None) or getattr(args, "topN", None) or getattr(args, "topn", None)
    # 与 parse_args 保持一致（若未传参，默认 10）
    topN = getattr(args, "top", None) or getattr(args, "top_n", None) or getattr(args, "topn", None) or getattr(args, "topN", None) or getattr(args, "top", 10)
    # 上面为兼容旧版参数，若你已在 parse_args 定义了 --top / --top_n / --topN 之一，可删掉上一行
    # 若已按前文提供的 parse_args 定义 --topN 或 --top10，请改成相应名字
    # 这里从 args 读取我们在新版 parse_args 中定义的 --topN 或 --top10 或 --top 参数
    top_key = "topN" if hasattr(args, "topN") else ("top10" if hasattr(args, "top10") else "top")
    top_k = getattr(args, "topN", getattr(args, "top10", getattr(args, "top", 10)))

    # 使用已实现的绘图函数：它会对 df 内部再次 groupby 取均值
    plot_global_top(df_agg.rename(columns={"S": "S"}), top_k, fig_dir / "seasonal_rank_global.png")

    # 6) 输出汇总表（宽表：行=模型，列=各区域 S，另附 S_mean）
    log("输出汇总表")
    wide = df_agg.pivot_table(index="model", columns="region", values="S", aggfunc="first")
    wide["S_mean"] = df_agg.groupby("model")["S"].mean()
    out_csv = tbl_dir / "seasonal_rank_summary.csv"
    wide.sort_values("S_mean", ascending=False).reset_index().to_csv(out_csv, index=False)
    log(f"完成：{out_csv}")

if __name__ == "__main__":
    main()
