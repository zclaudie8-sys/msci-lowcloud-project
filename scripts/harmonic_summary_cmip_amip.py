#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMIP6 AMIP harmonic summary vs observations (5x5 monthly)

功能：
- 遍历 /cmip6/5x5/<var> 目录下 *_amip_*.nc 文件（不同模型）
- 每个 模型×区域：区域均值 → 年份窗口 → 月气候态（12个月）→ 一阶谐波拟合
- 同口径加载观测（MODIS 或 LCRE）
- 产出：总表 CSV（tables/）+ 4 类图（figures/）

用法示例：
  python scripts/harmonic_summary_cmip_amip.py --var cllisccp
  python scripts/harmonic_summary_cmip_amip.py --var clswlow --regions SEP,SEA,NEP
  python scripts/harmonic_summary_cmip_amip.py --var cllisccp --start 2003 --end 2012
"""

from __future__ import annotations
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# --- 稳健导入：支持从项目根或任意位置运行 ---
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # scripts/
sys.path.insert(0, str(HERE.parent))   # project root

try:
    from scripts.utils_config import load_config, get_regions, setup_logger, make_output_path
    from scripts.utils_region import select_region_mean
    import scripts.utils_climatology as climu
except ModuleNotFoundError:
    from utils_config import load_config, get_regions, setup_logger, make_output_path
    from utils_region import select_region_mean
    import utils_climatology as climu

# CMIP 变量 ↔ 观测变量 映射
VAR_OBS_MAP = {
    "cllisccp": "cllmodis",  # 模式低云量 ↔ MODIS 低云量
    "clswlow":  "clswlow",   # SW LCRE
    "cllwlow":  "cllwlow",   # LW LCRE
}

def wrap_phase_diff(deg_model: float, deg_obs: float) -> float:
    """(model - obs) 的相位差，包裹到 [-180, 180]。"""
    return float((deg_model - deg_obs + 180.0) % 360.0 - 180.0)

def model_name_from_fname(fname: str) -> str:
    """从文件名推断模型名：优先取 'amip' 前一个 token；否则退化到第 3 个 token。"""
    base = Path(fname).name
    tokens = base.replace(".nc", "").split("_")
    for i, t in enumerate(tokens):
        if t.lower() == "amip" and i >= 1:
            return tokens[i - 1]
    return tokens[2] if len(tokens) >= 3 else base

def pearsonr12(a: xr.DataArray, b: xr.DataArray) -> float:
    """计算 12 个月气候态的皮尔森相关。"""
    av = a.values.astype(float)
    bv = b.values.astype(float)
    m = np.isfinite(av) & np.isfinite(bv)
    if m.sum() < 3:
        return np.nan
    av, bv = av[m], bv[m]
    if av.std() == 0 or bv.std() == 0:
        return np.nan
    return float(np.corrcoef(av, bv)[0, 1])

def _std12(arr) -> float:
    """std over 12-month climatology (ddof=1)."""
    v = np.asarray(arr.values if hasattr(arr, "values") else arr, dtype=float)
    m = np.isfinite(v)
    v = v[m]
    return float(np.std(v, ddof=1)) if v.size >= 2 else np.nan

def _corr12_centered(a, b) -> float:
    """Pearson r on 12-month climatology after removing their own annual means."""
    av = np.asarray(a.values if hasattr(a, "values") else a, dtype=float)
    bv = np.asarray(b.values if hasattr(b, "values") else b, dtype=float)
    m = np.isfinite(av) & np.isfinite(bv)
    av, bv = av[m], bv[m]
    if av.size < 3:
        return np.nan
    av -= av.mean()
    bv -= bv.mean()
    sa, sb = av.std(ddof=1), bv.std(ddof=1)
    if sa == 0 or sb == 0:
        return np.nan
    return float(np.corrcoef(av, bv)[0, 1])

def _reorder_box_to_WESN(box):
    """统一返回 [W, E, S, N]。"""
    if isinstance(box, dict):
        return [float(box["lon_min"]), float(box["lon_max"]),
                float(box["lat_min"]), float(box["lat_max"])]
    if isinstance(box, (list, tuple)) and len(box) == 4:
        b0, b1, b2, b3 = map(float, box)
        def _is_lat(x): return -90.0 <= x <= 90.0
        def _is_lon(x): return -360.0 <= x <= 360.0
        if _is_lon(b0) and _is_lon(b1) and _is_lat(b2) and _is_lat(b3):
            return [b0, b1, b2, b3]
        if _is_lat(b0) and _is_lat(b1) and _is_lon(b2) and _is_lon(b3):
            return [b2, b3, b0, b1]
    raise ValueError(f"无法识别的 region box 顺序: {box}")

def main():
    # ---------- 参数 ----------
    ap = argparse.ArgumentParser(description="CMIP6 AMIP harmonic summary vs observations.")
    ap.add_argument("--var", required=True, choices=list(VAR_OBS_MAP.keys()),
                    help="CMIP6 variable: cllisccp / clswlow / cllwlow")
    ap.add_argument("--regions", default="", help="Comma-separated region keys (e.g., SEP,SEA,NEP)")
    ap.add_argument("--cmip_path", default="", help="Override CMIP6 base path (optional)")
    ap.add_argument("--start", type=int, default=2003, help="Start year for comparison window")
    ap.add_argument("--end",   type=int, default=2014, help="End year for comparison window")
    args = ap.parse_args()

    # ---------- 配置 & 日志 ----------
    cfg = load_config("configs/config.yaml")
    logger = setup_logger(cfg, name="harmonic_summary_cmip_amip")

    # 比较窗口：默认 2003–2014，与项目总窗口取交集
    y0_cfg, y1_cfg = cfg["project"]["time_span"]
    y_start = max(y0_cfg, args.start)
    y_end   = min(y1_cfg, args.end)
    span_label = f"{y_start}-{y_end}"
    logger.info(f"Using comparison window: {span_label}")

    regions = [r.strip() for r in args.regions.split(",") if r.strip()] \
              if args.regions else cfg["project"]["regions"]
    logger.info(f"Regions: {regions}")

    # ---------- 路径 ----------
    cmip_base = Path(args.cmip_path or "/gws/nopw/j04/csgap/ceppi/data/cmip6/5x5/")
    var_dir = cmip_base / args.var
    if not var_dir.exists():
        raise FileNotFoundError(f"CMIP var dir not found: {var_dir}")

    obs_var = VAR_OBS_MAP[args.var]
    modis_dir = Path(cfg["data"]["modis_path"])
    obs_file = {
        "cllmodis": "cllmodis_200207-202406.nc",
        "clswlow":  "clswlow_200207-202406.nc",
        "cllwlow":  "cllwlow_200207-202406.nc",
    }[obs_var]
    obs_path = modis_dir / obs_file
    if not obs_path.exists():
        raise FileNotFoundError(f"OBS not found: {obs_path}")
    logger.info(f"OBS: {obs_var} -> {obs_path}")

    # ---------- 加载观测 ----------
    da_obs_full = xr.open_dataset(obs_path)[obs_var]

    # ---------- 找 AMIP 文件 ----------
    files = sorted(var_dir.glob("*amip*.nc"))
    if not files:
        raise FileNotFoundError(f"No AMIP files found under {var_dir} pattern '*amip*.nc'")
    logger.info(f"Found {len(files)} AMIP files")

    # ---------- 主循环：模型 × 区域 ----------
    records = []
    for f in files:
        model_id = model_name_from_fname(f.name)
        logger.info(f"[MODEL] {model_id} | file={f}")

        try:
            ds_mod = xr.open_dataset(f)
            if args.var not in ds_mod.data_vars:
                raise KeyError(f"Var '{args.var}' not in {f.name}; vars={list(ds_mod.data_vars)}")
            da_mod_full = ds_mod[args.var]
        except Exception as e:
            logger.exception(f"Skip {model_id} due to open/var error: {e}")
            continue

        for r in regions:
            raw_box = get_regions(cfg, [r])[r]
            box = _reorder_box_to_WESN(raw_box)

            # MODEL 区域均值 & 时间窗
            ts_mod = select_region_mean(da_mod_full, box)
            ts_mod = ts_mod.where(
                (ts_mod["time"].dt.year >= y_start) & (ts_mod["time"].dt.year <= y_end),
                drop=True
            )
            if ts_mod.size == 0:
                logger.warning(f"No MODEL data in span for model={model_id}, region={r}")
                continue

            # OBS 同口径
            ts_obs = select_region_mean(da_obs_full, box)
            ts_obs = ts_obs.where(
                (ts_obs["time"].dt.year >= y_start) & (ts_obs["time"].dt.year <= y_end),
                drop=True
            )
            if ts_obs.size == 0:
                logger.warning(f"No OBS data in span for region={r}")
                continue

            # 月气候态（1..12）
            clim_mod = ts_mod.groupby("time.month").mean("time", skipna=True).sel(month=np.arange(1, 13))
            clim_obs = ts_obs.groupby("time.month").mean("time", skipna=True).sel(month=np.arange(1, 13))

            # 一阶谐波拟合（统一软模式）
            fit_mod = climu.first_harmonic_fit(clim_mod, min_valid=8, soft=True)
            fit_obs = climu.first_harmonic_fit(clim_obs, min_valid=8, soft=True)

            A_m, phi_m, R2_m = fit_mod["amplitude"], fit_mod["phase_deg"], fit_mod["r2"]
            A_o, phi_o, R2_o = fit_obs["amplitude"], fit_obs["phase_deg"], fit_obs["r2"]
            dphi = wrap_phase_diff(phi_m, phi_o)
            Arat = (A_m / A_o) if (np.isfinite(A_o) and (A_o != 0)) else np.nan
            corr12 = pearsonr12(fit_mod["climatology"], fit_obs["climatology"])
            std_mod = _std12(fit_mod["climatology"])
            std_obs = _std12(fit_obs["climatology"])
            corr12_ctr = _corr12_centered(fit_mod["climatology"], fit_obs["climatology"])

            records.append({
                "model": model_id,
                "region": r,
                "var": args.var,
                "span": span_label,
                "A_model": float(A_m),
                "phi_model_deg": float(phi_m),
                "R2_model": float(R2_m),
                "A_obs": float(A_o),
                "phi_obs_deg": float(phi_o),
                "R2_obs": float(R2_o),
                "delta_phase_deg": float(dphi),
                "amplitude_ratio": float(Arat),
                "corr_clim12": float(corr12),
                "corr_clim12_centered": float(corr12_ctr),   # ✅ 新增：中心化相关
                "std_clim12_model": float(std_mod),          # ✅ 新增：σ_model
                "std_clim12_obs": float(std_obs),            # ✅ 新增：σ_obs
                "n_valid_model": float(fit_mod.get("n_valid", np.nan)),
                "n_valid_obs": float(fit_obs.get("n_valid", np.nan)),
            })

    if not records:
        raise RuntimeError("No records produced—check files, regions, or time window.")

    df = pd.DataFrame(records)

    # ---------- 保存总表（tables/） ----------
    out_csv = make_output_path(cfg, "tables", f"cmip_amip_{args.var}_vs_obs_{span_label}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved table: {out_csv}")

    # ---------- 按区域图（figures/）：比率、偏差、Δphase ----------
    for r in sorted(df["region"].unique()):
        sub = df[df["region"] == r].copy()
        if len(sub) == 0:
            continue

        # 计算比率/偏差
        obs_val = float(sub["A_obs"].iloc[0]) if len(sub) else np.nan
        sub["A_ratio"] = (sub["A_model"] / obs_val) if (np.isfinite(obs_val) and obs_val != 0) else np.nan
        sub["A_bias"]  = sub["A_model"] - obs_val

        # (1) 比率
        fig1 = make_output_path(cfg, "figures", f"cmip_amip_{args.var}_A_ratio_{r}_{span_label}.png")
        plt.figure(figsize=(9, 4.0))
        order = np.argsort(sub["A_ratio"].values)
        xs = np.arange(len(order))
        plt.scatter(xs, sub["A_ratio"].values[order])
        for i, (_, row) in enumerate(sub.iloc[order].iterrows()):
            plt.text(xs[i], row["A_ratio"], row["model"], fontsize=7,
                     rotation=45, ha="right", va="bottom", alpha=0.85)
        plt.axhline(1.0, ls="--", c="k", lw=1)
        plt.ylabel("Amplitude ratio (MODEL / OBS)")
        plt.title(f"{args.var} | Amplitude ratio | {r} | {span_label}")
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(fig1, dpi=200, bbox_inches="tight")
        plt.close()

        # (2) 偏差
        fig1b = make_output_path(cfg, "figures", f"cmip_amip_{args.var}_A_bias_{r}_{span_label}.png")
        plt.figure(figsize=(9, 4.0))
        order = np.argsort(sub["A_bias"].values)
        xs = np.arange(len(order))
        plt.scatter(xs, sub["A_bias"].values[order])
        for i, (_, row) in enumerate(sub.iloc[order].iterrows()):
            plt.text(xs[i], row["A_bias"], row["model"], fontsize=7,
                     rotation=45, ha="right", va="bottom", alpha=0.85)
        plt.axhline(0.0, ls="--", c="k", lw=1)
        plt.ylabel("Amplitude bias (MODEL − OBS)")
        plt.title(f"{args.var} | Amplitude bias | {r} | {span_label}")
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(fig1b, dpi=200, bbox_inches="tight")
        plt.close()

        # (3) Δphase
        fig2 = make_output_path(cfg, "figures", f"cmip_amip_{args.var}_dPhase_{r}_{span_label}.png")
        plt.figure(figsize=(8.5, 4.2))
        sub_sorted = sub.sort_values("delta_phase_deg")
        plt.bar(sub_sorted["model"], sub_sorted["delta_phase_deg"])
        plt.axhline(0, linewidth=1, color="k")
        plt.ylabel("Δphase (deg) = model − obs")
        plt.xticks(rotation=60, ha="right")
        plt.title(f"{args.var} | Δphase MODEL−OBS | {r} | {span_label}")
        plt.tight_layout()
        plt.savefig(fig2, dpi=200, bbox_inches="tight")
        plt.close()

    # ---------- 跨区域合并散点（figures/） ----------
    fig3 = make_output_path(cfg, "figures", f"cmip_amip_{args.var}_A_vs_obs_multi_region_{span_label}.png")
    plt.figure(figsize=(7.8, 6.2))
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    for i, r in enumerate(sorted(df["region"].unique())):
        sub = df[df["region"] == r]
        if len(sub) == 0:
            continue
        plt.scatter(sub["A_obs"], sub["A_model"],
                    label=r, marker=markers[i % len(markers)], alpha=0.9, edgecolors="none")

    # 1:1 线与轴限
    max_obs = float(np.nanmax(df["A_obs"].values)) if np.isfinite(df["A_obs"]).any() else 1.0
    max_mod = float(np.nanmax(df["A_model"].values)) if np.isfinite(df["A_model"]).any() else 1.0
    lim = max(max_obs, max_mod)
    lim = 1.0 if (not np.isfinite(lim) or lim <= 0) else lim * 1.1
    plt.plot([0, lim], [0, lim], "k--", lw=1)

    plt.xlabel("Amplitude OBS")
    plt.ylabel("Amplitude MODEL")
    plt.title(f"{args.var} | Amplitude: MODEL vs OBS | all regions | {span_label}")
    plt.legend(title="Region", frameon=False, ncol=3)
    plt.tight_layout()
    plt.savefig(fig3, dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
