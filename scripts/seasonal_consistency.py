#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seasonal consistency diagnostics (Shape / Amplitude / Phase) + Taylor diagram.

Metrics per model (within a region):
- Shape correlation r:
    * raw   : Pearson r between 12-month climatologies (from tables file)
    * harm  : correlation of first-harmonic reconstructions -> r_harm = cos(Δφ)
- Amplitude:
    * std_ratio ≈ σ_model / σ_obs      (proxy via harmonic amplitude ratio if std not available)
    * A_ratio   = A_model / A_obs      (from tables file)
- Phase:
    * dphi = φ_model − φ_obs normalized to [−180°, 180°)

Inputs:
  - tables/cmip_amip_{var}_vs_obs_{span}.csv   (produced by harmonic_summary_cmip_amip.py)
    required columns: ['model','region','amplitude_ratio','delta_phase_deg','corr_clim12',
                       'A_model','A_obs','phi_model_deg','phi_obs_deg']

Usage:
  python scripts/seasonal_consistency.py --var cllisccp --region NEP --shape raw
  python scripts/seasonal_consistency.py --var clswlow  --regions ALL --shape harm

Outputs:
  - output/tables/SEASONAL_consistency_{var}_{region}.csv        (per-model metrics)
  - output/figures/Fig_Taylor_{var}_{region}.png                  (Taylor diagram)
  - If --regions ALL: one CSV per region + one Taylor per region.
"""

from __future__ import annotations
from pathlib import Path
import sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# robust imports
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
try:
    from scripts.utils_config import load_config, make_output_path, setup_logger
except ModuleNotFoundError:
    from utils_config import load_config, make_output_path, setup_logger


# -------------------- ARGS --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Seasonal consistency stats + Taylor diagram.")
    p.add_argument("--var", required=True, help="cllisccp / clswlow / cllwlow")
    p.add_argument("--region", help="Single region key (NEP/NEA/SEP/SEA/SEI)")
    p.add_argument("--regions", default="", help="Comma list or ALL to use project.regions")
    p.add_argument("--span", default="", help="Time span tag used in the tables file, e.g., 2003-2014 (optional)")
    p.add_argument("--shape", choices=["raw","harm"], default="raw",
                   help="Shape correlation: 'raw' (12-month r) or 'harm' (cos Δφ). Default raw.")
    return p.parse_args()


# -------------------- HELPERS --------------------
def _regions_to_run(cfg, args):
    if args.regions:
        rs = [r.strip() for r in args.regions.split(",") if r.strip()]
        if len(rs) == 1 and rs[0].upper() == "ALL":
            rs = cfg["project"]["regions"]
        return rs
    if args.region:
        return [args.region]
    raise SystemExit("请提供 --region 或 --regions（可用 ALL）")

def _wrap_dphi(deg):
    # normalize to [-180,180)
    return (deg + 180.0) % 360.0 - 180.0

def _load_table(cfg, var, span):
    tab_dir = Path(cfg["output"]["root"]) / cfg["output"]["subdirs"]["tables"]
    if span:
        fname = f"cmip_amip_{var}_vs_obs_{span}.csv"
    else:
        # fallback: pick the newest matching file
        cands = sorted(tab_dir.glob(f"cmip_amip_{var}_vs_obs_*.csv"))
        if not cands:
            raise FileNotFoundError(f"No tables for var={var} under {tab_dir}")
        fname = cands[-1].name
    df = pd.read_csv(tab_dir / fname)
    # sanity columns
    need = ["model","region","amplitude_ratio","delta_phase_deg","corr_clim12",
            "A_model","A_obs","phi_model_deg","phi_obs_deg"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {fname}: {missing}")
    return df, fname

def _build_metrics(df_region: pd.DataFrame, shape_mode: str) -> pd.DataFrame:
    out_rows = []
    for _, r in df_region.iterrows():
        model = r["model"]
        # phase
        dphi = _wrap_dphi(float(r["delta_phase_deg"]))
        # amplitude
        A_ratio = safe_div(r["A_model"], r["A_obs"])
        # std ratio (Taylor): if monthly stds unavailable, we proxy with A_ratio
        std_ratio = A_ratio
        # shape correlation
        if shape_mode == "raw":
            shape_r = float(r["corr_clim12"])
        else:
            # for pure sinusoids, r_harm = cos(Δφ)
            shape_r = float(np.cos(np.deg2rad(dphi)))
        out_rows.append({
            "model": model,
            "shape_r": shape_r,
            "std_ratio": std_ratio,
            "A_ratio": A_ratio,
            "dphi_deg": dphi,
            "A_model": float(r["A_model"]),
            "A_obs": float(r["A_obs"]),
            "phi_model_deg": float(r["phi_model_deg"]),
            "phi_obs_deg": float(r["phi_obs_deg"]),
        })
    return pd.DataFrame(out_rows)

def safe_div(a, b):
    a = float(a); b = float(b)
    return np.nan if (not np.isfinite(b) or b == 0.0) else a / b

'''
def _pick_key_indices(A_ratio, dphi_deg, r_shape, k_total=12):
    """
    选出最有信息量的点：|A×-1| 最大、|Δφ| 最大、r 最低。
    返回要标注的索引（去重、排序）。
    """
    A_ratio = np.asarray(A_ratio, float)
    dphi    = np.asarray(dphi_deg, float)
    r       = np.asarray(r_shape, float)

    k1 = max(3, k_total // 2)          # 振幅离群
    k2 = max(3, k_total // 3)          # 相位离群
    k3 = max(2, k_total - k1 - k2)     # 相关最低

    idx_amp = np.argsort(np.abs(A_ratio - 1.0))[-k1:]
    idx_phi = np.argsort(np.abs(dphi))[-k2:]
    idx_r   = np.argsort(r)[:k3]

    idx = sorted(set(idx_amp.tolist() + idx_phi.tolist() + idx_r.tolist()))
    return idx

def _annotate_keypoints(ax, theta, rho, names, A_ratio, dphi_deg, idx_to_label):
    """在极坐标上只标注 idx_to_label 指定的点；带白描边，轻微错位避免重叠。"""
    rng = np.random.default_rng(42)
    for i in idx_to_label:
        th = theta[i] + np.deg2rad(rng.uniform(-1.2, 1.2))
        rh = rho[i]   + 0.06 + rng.uniform(-0.02, 0.02)
        lbl = f"{names[i]}"
        txt = ax.text(th, rh, lbl, fontsize=7.5, ha="center", va="bottom", color="#111")
        txt.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])

def _annotate_keypoints_side(ax, theta, rho, names, A_ratio, dphi_deg,
                             idx_to_label, side="east", rho_max=1.6,
                             min_gap=0.05):
    """
    将关键点的标签统一放到一侧（east/west），并用引线连接。
    - side: "east" 右侧（θ≈0°），"west" 左侧（θ≈π）
    - min_gap: 侧边标签在径向方向的最小间距（极坐标半径单位）
    """
    import matplotlib.patheffects as pe
    rng = np.random.default_rng(7)

    # 侧边的目标角度（右侧=0°, 左侧=π）
    theta_lbl = 0.02 if side == "east" else (np.pi - 0.02)

    # 先按原始 rho 排序，便于从上到下（或下到上）排布，减少交叉
    order = np.argsort(rho[idx_to_label])[::-1]  # 从大到小
    idx_sorted = [idx_to_label[i] for i in order]

    # 目标半径初值：映射到 [0.15, rho_max*0.98] 区间，再做避让
    rmin, rmax = 0.15, rho_max*0.98
    rho_src = rho[idx_sorted]
    rho_tgt = np.interp(rho_src, [rho.min(), rho.max()], [rmin, rmax])

    # 简单避让：自上而下保证相邻至少 min_gap
    for i in range(1, len(rho_tgt)):
        if rho_tgt[i-1] - rho_tgt[i] < min_gap:
            rho_tgt[i] = rho_tgt[i-1] - min_gap
    # 轻微随机扰动，减少完全等距的视觉“叠印”
    rho_tgt += rng.uniform(-0.01, 0.01, size=len(rho_tgt))

    # 画引线 + 标签（标签放侧边；引线连接到真实点）
    for j, i in enumerate(idx_sorted):
        # 引线：点 -> 侧边标签锚点
        ax.annotate("",
            xy=(theta[i], rho[i]),
            xytext=(theta_lbl, rho_tgt[j]),
            textcoords="data",
            arrowprops=dict(arrowstyle="-", lw=0.8, color="#555",
                            shrinkA=0, shrinkB=0, alpha=0.9,
                            connectionstyle="arc3,rad=0.0"),
            zorder=3)

        # 标签文本
        lbl = f"{names[i]}"
        txt = ax.text(theta_lbl, rho_tgt[j], lbl,
                      fontsize=7.8, ha=("left" if side=="east" else "right"),
                      va="center", color="#111")
        txt.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white")])
'''

# -------------------- TAYLOR DIAGRAM --------------------

import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm

def plot_taylor(ax, metrics, title: str, rho_max=1.6,
                r_levels=(0.2, 0.4, 0.6, 0.8, 0.9, 0.95),
                std_levels=None,
                point_size=42):
    """
    最终标准版 Taylor diagram：
    - 背景：灰色同心圆 + 相关射线（0–180°）
    - 内容：散点 + Δphase 色标
    - 无任何文字标注或引线
    """

    # 取数据
    r    = np.clip(metrics["shape_r"].values.astype(float), -1, 1)
    rho  = np.clip(metrics["std_ratio"].values.astype(float), 0, rho_max)
    dphi = metrics["dphi_deg"].values.astype(float)
    theta = np.arccos(r)

    # -------------------- 背景 --------------------
    # (1) 相关系数射线
    for rc in r_levels:
        th = np.arccos(np.clip(rc, -1, 1))
        ax.plot([th, th], [0, rho_max], color="#c9c9c9", lw=0.8, alpha=0.8, zorder=0)
        ax.text(th, rho_max * 1.02, f"{rc:.2f}", ha="center", va="bottom",
                fontsize=8, color="#7a7a7a")

    # (2) 同心圆（标准差比刻度）
    if std_levels is None:
        std_levels = np.arange(0.2, rho_max + 0.001, 0.2)
    t = np.linspace(0, np.pi, 361)
    for s in std_levels:
        ax.plot(t, np.full_like(t, s), color="#d3d3d3", lw=0.6, alpha=0.8, zorder=0)
    # σ=1 加粗
    ax.plot(t, np.ones_like(t), color="#6b6b6b", lw=1.3, alpha=0.95, zorder=0)

    # (3) 极坐标刻度
    ax.set_thetagrids(np.arange(0, 181, 30),
                      labels=[f"{ang}°" for ang in np.arange(0, 181, 30)],
                      fontsize=8)
    ax.set_rgrids(std_levels, angle=150,
                  labels=[f"{v:.2f}" for v in std_levels],
                  color="#666", fontsize=8)

    # -------------------- 散点 --------------------
    norm = TwoSlopeNorm(vmin=-180, vcenter=0, vmax=180)
    cmap = plt.get_cmap("coolwarm")
    sc = ax.scatter(theta, rho, c=dphi, cmap=cmap, norm=norm,
                    s=point_size, alpha=0.9, edgecolors="k", linewidths=0.3, zorder=3)

    # -------------------- 轴样式 --------------------
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_thetalim(0, np.pi)
    ax.set_rlim(0, rho_max * 1.05)
    ax.grid(False)
    ax.set_title(title, fontsize=12, pad=10)

    # -------------------- 色条 --------------------
    cbar = plt.colorbar(sc, ax=ax, pad=0.08, fraction=0.05)
    cbar.set_label("Δphase (deg)", fontsize=9)
    cbar.set_ticks([-180, -120, -60, 0, 60, 120, 180])
    for lab in cbar.ax.yaxis.get_ticklabels():
        lab.set_fontsize(8)



# -------------------- MAIN --------------------

def main():
    args   = parse_args()
    cfg    = load_config(str(ROOT / "configs" / "config.yaml"))
    logger = setup_logger(cfg, name="seasonal_consistency")

    regions      = _regions_to_run(cfg, args)
    df_all, src  = _load_table(cfg, args.var, args.span)
    logger.info(f"Using tables file: {src}")

    for region in regions:
        sub = df_all[df_all["region"] == region].copy()
        if sub.empty:
            logger.warning(f"[{region}] no rows in {src}, skip.")
            continue

        # 计算 per-model 指标（r/std_ratio/A_ratio/dphi…）
        met = _build_metrics(sub, args.shape)



        '''
        # 导出：全点表（含是否标注 & 排序指标）
        met_out = met.copy()
        met_out.insert(0, "region", region)
        met_out["is_annotated"]   = False
        met_out["rank_abs_Adiff"] = met_out["A_ratio"].sub(1).abs().rank(method="min")
        met_out["rank_abs_dphi"]  = met_out["dphi_deg"].abs().rank(method="min", ascending=False)
        met_out["rank_low_r"]     = met_out["shape_r"].rank(method="min")  # r 越小越靠前
        '''
        csv_all = make_output_path(cfg, "tables", f"Taylor_points_{args.var}_{region}.csv")
        '''
                
        met_out.to_csv(csv_all, index=False)
        '''

        logger.info(f"[{region}] saved points table → {csv_all}")

        # 导出：仅关键点子表（论文附表更好排版）
        csv_lab = make_output_path(cfg, "tables", f"Taylor_points_labeled_{args.var}_{region}.csv")
        logger.info(f"[{region}] saved labeled subset → {csv_lab}")

        # 画图（只标关键点）
        fig = plt.figure(figsize=(8.2, 6.6))
        ax  = plt.subplot(111, projection="polar")
        title = f"Taylor — {args.var} | {region} | r={args.shape}"
        plot_taylor(ax, met, title, rho_max=1.6)
        fig_path = make_output_path(cfg, "figures", f"Fig_Taylor_{args.var}_{region}.png")
        plt.tight_layout(); plt.savefig(fig_path, dpi=400, bbox_inches="tight"); plt.close()

        logger.info(f"[{region}] saved Taylor diagram → {fig_path}")

    logger.info("DONE.")


if __name__ == "__main__":
    main()
