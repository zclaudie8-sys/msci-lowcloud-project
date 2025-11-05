#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harmonic_yearly_obs.py

逐年一阶谐波拟合（MODIS obs）→ 汇总每年的振幅 A_y 与相位 φ_y，并画相位玫瑰图与年度振幅折线。

Outputs:
  - output/harmonics/<var>_yearly_harmonics_<span>.csv
      columns: region, year, n_months, amplitude, phase_deg, peak_month, r2
  - figures/harmonics/<var>_phase_rose_<region>_<span>.png
  - figures/harmonics/<var>_amp_timeseries_<region>_<span>.png

Usage:
  python scripts/harmonic_yearly_obs.py --var cllmodis
  # options: --regions NEP,NEA,SEA,SEI,SEP  --min_months 8
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# --- project utils (与现有脚本一致) ---
from utils_config import load_config, get_regions, setup_logger, make_output_path
from utils_region import select_region_mean

# 尝试使用已有的一阶谐波函数；若不存在则提供本地实现
try:
    from utils_climatology import first_harmonic_fit  # 接受 12×1 的月序列
except Exception:
    def first_harmonic_fit(monthly_values: np.ndarray | pd.Series) -> dict:
        """最小二乘拟合：x_m = c + a cos(2πm/12) + b sin(2πm/12)"""
        arr = np.asarray(monthly_values, float)
        # 允许 NaN：仅用有效月
        months = np.array([m for m in range(1,13) if np.isfinite(arr[m-1])], int)
        y = np.array([arr[m-1] for m in months], float)
        if len(months) < 2:
            return dict(amplitude=np.nan, phase_deg=np.nan, peak_month=np.nan, r2=np.nan)
        omega = 2*np.pi/12.0
        X = np.column_stack([np.ones(len(months)),
                             np.cos(omega*months),
                             np.sin(omega*months)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        c, a, b = beta
        yhat = X @ beta
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - y.mean())**2)) if len(y) > 1 else np.nan
        r2 = np.nan if not np.isfinite(ss_tot) or ss_tot == 0 else 1 - ss_res/ss_tot
        amp = float(np.hypot(a, b))
        phase = float(np.degrees(np.arctan2(b, a)))  # [-180,180]
        if phase < 0: phase += 360.0
        peak_month = int(np.round(((phase/360.0) * 12) % 12)); peak_month = 12 if peak_month==0 else peak_month
        return dict(amplitude=amp, phase_deg=phase, peak_month=peak_month, r2=r2)

VAR_TO_FILE = {
    "cllmodis": "cllmodis_200207-202406.nc",
    "clswlow":  "clswlow_200207-202406.nc",
    "cllwlow":  "cllwlow_200207-202406.nc",
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Year-by-year first-harmonic summary for MODIS observations.")
    p.add_argument("--var", default="cllmodis", choices=list(VAR_TO_FILE.keys()))
    p.add_argument("--regions", default="", help="Comma-separated region keys (default=all from config).")
    p.add_argument("--min_months", type=int, default=8, help="每年拟合所需的最少有效月份数（默认8）")
    return p.parse_args()
def _reorder_box_to_WESN(box):
    """把 region box 统一成 [W, E, S, N]（lon_min, lon_max, lat_min, lat_max）。"""
    if isinstance(box, dict):
        return [float(box["lon_min"]), float(box["lon_max"]),
                float(box["lat_min"]), float(box["lat_max"])]
    if isinstance(box, (list, tuple)) and len(box) == 4:
        b0, b1, b2, b3 = map(float, box)
        def _is_lat(x): return -90.0 <= x <= 90.0
        def _is_lon(x): return -360.0 <= x <= 360.0
        # [lon, lon, lat, lat]
        if _is_lon(b0) and _is_lon(b1) and _is_lat(b2) and _is_lat(b3):
            return [b0, b1, b2, b3]
        # [lat, lat, lon, lon]
        if _is_lat(b0) and _is_lat(b1) and _is_lon(b2) and _is_lon(b3):
            return [b2, b3, b0, b1]
    raise ValueError(f"无法识别的 region box 顺序: {box}")

def _ensure_lon_lat_names(da):
    """兼容 lat|latitude、lon|longitude；返回 (lat_name, lon_name)。"""
    lat_name = "lat" if "lat" in da.coords else ("latitude" if "latitude" in da.coords else None)
    lon_name = "lon" if "lon" in da.coords else ("longitude" if "longitude" in da.coords else None)
    if lat_name is None or lon_name is None:
        raise KeyError(f"未找到经纬坐标（lat/latitude, lon/longitude）")
    return lat_name, lon_name

def region_mean_robust(da, box):
    """
    稳健区域平均：
    - 统一经度到 [-180,180)
    - 自动适配经纬轴名和升/降序
    - 用 skipna=True 做面积平均
    输入 box 可为 [lat_min, lat_max, lon_min, lon_max] 或 [lon_min, lon_max, lat_min, lat_max] 或 dict。
    返回：time 序列（DataArray）
    """
    # 取轴名
    lat_name, lon_name = _ensure_lon_lat_names(da)

    # 经度 0–360 -> [-180,180)
    lon = da[lon_name]
    if float(lon.max()) > 180.0:
        da = da.assign_coords({lon_name: (((lon + 180.0) % 360.0) - 180.0)}).sortby(lon_name)

    # 统一到 WESN
    W, E, S, N = _reorder_box_to_WESN(box)

    # 处理 lat 升/降序
    lat_vals = da[lat_name].values
    lat_slice = slice(S, N) if lat_vals[0] < lat_vals[-1] else slice(N, S)

    # 经度不跨日界线（你的 NEP/SEI 也不跨），直接 slice
    lon_vals = da[lon_name].values
    lon_slice = slice(W, E) if lon_vals[0] < lon_vals[-1] else slice(E, W)

    # 切区域
    da_reg = da.sel({lat_name: lat_slice, lon_name: lon_slice})

    # 面积加权（cosφ）可选：需要时可解注释
    # weights = np.cos(np.deg2rad(da_reg[lat_name]))
    # ts = da_reg.weighted(weights).mean(dim=[lat_name, lon_name], skipna=True)

    # 简单均值
    ts = da_reg.mean(dim=[lat_name, lon_name], skipna=True)
    return ts

def rayleigh_stats(phases_deg: np.ndarray) -> tuple[float,float,float,float,float]:
    """Rayleigh 圆统计：返回 (N, R, z, p, mean_phase_deg)"""
    phi = phases_deg[np.isfinite(phases_deg)]
    # 统一到 [0,360)
    phi = (phi % 360.0 + 360.0) % 360.0
    th  = np.radians(phi)

    n = len(th)
    if n == 0:
        return 0.0, np.nan, np.nan, np.nan, np.nan

    C, S = np.sum(np.cos(th)), np.sum(np.sin(th))
    R = np.hypot(C, S) / n
    z = n * R * R

    # p 值近似（Berens 2009, CircStat）
    p = np.exp(-z) * (1 + (2*z - z*z)/(4*n) - (24*z - 132*z*z + 76*z**3 - 9*z**4)/(288*n*n))
    p = float(np.clip(p, 0.0, 1.0))

    mean_rad = np.arctan2(S, C)
    if mean_rad < 0:
        mean_rad += 2*np.pi
    mean_deg = float(np.degrees(mean_rad))

    return float(n), float(R), float(z), p, mean_deg


'''
def phase_rose(phases_deg: np.ndarray, region: str, span: str, out_png: Path, bins: int = 12):
    th = np.radians(phases_deg[np.isfinite(phases_deg)])
    if th.size == 0:
        return
    counts, edges = np.histogram(th, bins=bins, range=(0, 2*np.pi))
    widths = np.diff(edges)
    centers = edges[:-1] + widths/2
    N, R, z, p = rayleigh_stats(phases_deg)
    mean_dir = np.degrees(np.arctan2(np.sum(np.sin(th)), np.sum(np.cos(th))))
    if mean_dir < 0: mean_dir += 360.0

    fig = plt.figure(figsize=(6.4, 6.4))
    ax = plt.subplot(111, polar=True)
    ax.bar(centers, counts, width=widths, align="center", edgecolor="white", alpha=0.85)
    rmax = counts.max() if counts.max()>0 else 1
    ax.annotate("", xy=(np.radians(mean_dir), R*rmax),
                xytext=(np.radians(mean_dir), 0.0),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"Phase rose — {region} | {span}\nRayleigh: R={R:.2f}, p={p:.3g}, mean={mean_dir:.0f}°",
                 fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
'''

def plot_phase_rose(phases_deg, R, p, mean_deg, region, out_png, bins=12):
    phi = (phases_deg[np.isfinite(phases_deg)] % 360.0 + 360.0) % 360.0
    theta = np.radians(phi)
    if theta.size == 0:
        print(f"[{region}] 无有效相位，跳过画图。")
        return

    counts, edges = np.histogram(theta, bins=bins, range=(0, 2*np.pi))
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2

    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(3.8, 3.8))
    ax = plt.subplot(111, polar=True)

    # 主体直方图
    bars = ax.bar(centers, counts, width=widths, align="center",
                  edgecolor="none", color="#1f77b4", alpha=0.8)

    # 平均相位箭头
    mean_rad = np.deg2rad(mean_deg)
    rmax = counts.max() if counts.max() > 0 else 1
    ax.annotate("",
                xy=(mean_rad, rmax * 0.95 * R),
                xytext=(mean_rad, 0),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="k"))

    # 格式与标签
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(range(0, 360, 45), fontsize=8)
    ax.set_rgrids([], [])  # 隐藏径向网格标签

    # 美化环形边框
    ax.spines["polar"].set_visible(False)

    ax.set_title(f"Phase rose — {region}\nR={R:.2f}, p={p:.1e}, mean={mean_deg:.0f}°",
                 fontsize=9, weight="bold", pad=10)
    plt.tight_layout(pad=0.1)
    fig.savefig(out_png.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)



def amp_timeseries(years, amps, region, span, out_png, var):
    years = np.asarray(years, int)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(years, amps, marker="o", markersize=5, lw=1.5, color="#1f77b4")

    # 清晰刻度与边界
    ax.set_xlim(years.min(), years.max())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(np.arange(years.min(), years.max() + 1, 2))  # 每2年一刻度
    ax.tick_params(labelsize=9)

    # 标签与标题
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Amplitude (% cloud)", fontsize=10)
    ax.set_title(f"{var.upper()} yearly amplitude — {region} ({span})",
                 fontsize=11, pad=6, weight="bold")

    # 美化边框与网格
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(alpha=0.3, lw=0.6)

    # 紧凑布局
    plt.tight_layout(pad=0.3)
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
    fig.savefig(out_png.with_suffix(".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)

def main():
    args = parse_args()

    # 1) 配置
    cfg = load_config("configs/config.yaml")
    logger = setup_logger(cfg, name="harmonic_yearly_obs")
    y0, y1 = cfg["project"]["time_span"]
    all_regions = cfg["project"]["regions"]
    regions = [r.strip() for r in args.regions.split(",") if r.strip()] if args.regions else all_regions
    span = f"{y0}-{y1}"
    logger.info(f"Time span: {span}")
    logger.info(f"Regions: {regions}")
    logger.info(f"Min months per year: {args.min_months}")

    # 2) 数据
    fpath = Path(cfg["data"]["modis_path"]) / VAR_TO_FILE[args.var]
    if not fpath.exists():
        raise FileNotFoundError(f"File not found: {fpath}")
    logger.info(f"Loading {args.var} from: {fpath}")
    ds = xr.open_dataset(fpath)
    if args.var not in ds.data_vars:
        raise KeyError(f"Variable '{args.var}' not in dataset. Found: {list(ds.data_vars)}")
    da_full = ds[args.var]
    # --- 统一经度到 [-180, 180]，以匹配你的区域框定义 ---
    if "lon" in da_full.coords:
        lon = da_full["lon"]
        if float(lon.max()) > 180.0:
            # 0~360 -> [-180,180)
            da_full = da_full.assign_coords(lon=(((lon + 180.0) % 360.0) - 180.0)).sortby("lon")

    # 3) 遍历区域与逐年拟合
    rows = []
    for r in regions:
        box = get_regions(cfg, [r])[r]
        logger.info(f"Region {r}: {box}")
        raw_box = get_regions(cfg, [r])[r]
        ts = region_mean_robust(da_full, raw_box).sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
        
        # 监控 NaN 比例（你之前的日志行保留/更新）
        nan_ratio = float(np.isnan(ts.values).mean())
        logger.info(f"[{r}] NaN ratio in selected time series (robust): {nan_ratio:.2%}")


        # 按年分组，每年内按 calendar month 聚合（防止同月多样本）
        df = ts.to_dataframe(name="val").reset_index()
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month
        yearly = df.groupby(["year","month"], as_index=False)["val"].mean()
        years = sorted(yearly["year"].unique())

        for y in years:
            ym = yearly[yearly["year"]==y]
            # 组装 12×1 月序列（允许缺测）
            arr = np.full(12, np.nan, float)
            arr[ym["month"].values.astype(int)-1] = ym["val"].values.astype(float)

            n_months = int(np.isfinite(arr).sum())
            if n_months < args.min_months:
                rows.append(dict(region=r, year=int(y), n_months=n_months,
                                 amplitude=np.nan, phase_deg=np.nan, peak_month=np.nan, r2=np.nan))
                continue


            # 将 numpy 数组 arr 包装为带 month 坐标的 DataArray，以兼容 utils_climatology 版本
            da = xr.DataArray(arr, coords={"month": np.arange(1, 13)}, dims="month")
            fit = first_harmonic_fit(da)
            phi = float(fit["phase_deg"])
            phi = (phi % 360.0 + 360.0) % 360.0
            rows.append(dict(region=r, year=int(y), n_months=n_months,
                 amplitude=float(fit["amplitude"]),
                 phase_deg=phi,  # 用规范后的相位
                 peak_month=int(fit["peak_month"]),
                 r2=float(fit["r2"])))


        # 区域级可视化（仅用有效年的相位/振幅）
        reg_df = pd.DataFrame([row for row in rows if row["region"]==r])
        valid = reg_df[np.isfinite(reg_df["phase_deg"])]
        print(f"[{r}] valid years for rose: {len(valid)} / {reg_df.shape[0]}")

        if not valid.empty:
            # 相位玫瑰
            # 先生成根目录 (figures/obsrose)
            rose_dir = make_output_path(cfg, "figures", "obsrose")
            rose_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
            
            # 再拼完整文件路径
            rose_png = rose_dir / f"{args.var}_phase_rose_{r}_{span}.png"
            amp_png  = rose_dir / f"{args.var}_amp_timeseries_{r}_{span}.png"
                        # 所有玫瑰图输出到 output/figures/obsrose/
            N, R, z, p, mean_deg = rayleigh_stats(valid["phase_deg"].to_numpy())

            plot_phase_rose(valid["phase_deg"].to_numpy(), R, p, mean_deg, r, rose_png, bins=12)
            # 振幅时序
            amp_timeseries(valid["year"].to_numpy(), valid["amplitude"].to_numpy(), r, span, amp_png, args.var)

    # 4) 保存汇总 CSV
    harmonics_dir = Path("output/harmonics")
    harmonics_dir.mkdir(parents=True, exist_ok=True)
    
    out_csv = harmonics_dir / f"{args.var}_yearly_harmonics_{span}.csv"
    pd.DataFrame(rows).sort_values(["region","year"]).to_csv(out_csv, index=False)
    logger.info(f"Saved yearly harmonics CSV → {out_csv}")
    logger.info("DONE.")

if __name__ == "__main__":
    main()
