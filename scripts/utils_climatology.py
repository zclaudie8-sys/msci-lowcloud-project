from __future__ import annotations
from typing import Optional, Dict
import numpy as np
import xarray as xr

from utils_region import select_region_mean  # 复用你的区域加权函数

from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

    
def regional_monthly_climatology(
    da: xr.DataArray,
    box,
    lat_name: Optional[str] = None,
    lon_name: Optional[str] = None,
    time_name: str = "time",
) -> xr.DataArray:
    """
    对传入的月度 DataArray 先做区域加权平均，再按 month 求多年平均，返回 12 个月的气候态。
    - da: 必须含 time 维，且为月频（可 .dt.month）
    - box: [W, E, S, N] 或你自定义的区域对象（utils_region 支持的那种）
    """
    lat_name = lat_name or "lat"
    lon_name = lon_name or "lon"
    # 1) 区域面积加权均值的时间序列
    ts = select_region_mean(da, box, lat_name=lat_name, lon_name=lon_name, time_name=time_name)
    print("months in ts:", np.unique(ts['time'].dt.month.values))   # 应有 1..12
    print("per-month counts:", ts.groupby("time.month").count().values) 
    if time_name not in ts.coords:
        raise ValueError("输入没有时间维，无法计算月气候态。")

    # 2) 按月取多年均值（12 个值）
    clim = ts.groupby(f"{time_name}.month").mean(dim=time_name, skipna=True)
    # 明确保证 month=1..12 的顺序
    clim = clim.sel(month=np.arange(1, 13))

    # 传承/更新元数据
    clim.attrs.update(ts.attrs)
    long_name = clim.attrs.get("long_name", "").strip()
    clim.attrs["long_name"] = (long_name + " monthly climatology").strip()
    return clim


def first_harmonic_fit(
    ts: xr.DataArray,
    assume_monthly: bool = True,
    time_name: str = "time",
    min_valid: int = 8,      # 拟合所需的“建议最少有效月份”
    soft: bool = False,      # True 时：有效月不足也不抛错，尽量拟合或返回 NaN
) -> Dict[str, object]:
    # 1) 取到 12 个月序列
    if "month" in ts.coords:
        y = ts.sel(month=np.arange(1, 13))
    else:
        if not (assume_monthly and time_name in ts.coords):
            raise ValueError("请提供 month 维的 climatology，或含 time 维的月序列（assume_monthly=True）。")
        y = ts.groupby(f"{time_name}.month").mean(dim=time_name, skipna=True)
        y = y.sel(month=np.arange(1, 13))

    y_values = y.values.astype(float)
    mask = np.isfinite(y_values)
    n_valid = int(mask.sum())

    def _empty_result():
        recon = xr.DataArray(np.full(12, np.nan, float), coords={"month": np.arange(1,13)}, dims=("month",))
        return {
            "a": np.nan, "b": np.nan, "c": np.nan,
            "amplitude": np.nan, "phase_rad": np.nan, "phase_deg": np.nan,
            "peak_month": np.nan, "trough_month": np.nan, "r2": np.nan,
            "recon": recon, "climatology": y, "n_valid": n_valid, "low_valid": True,
        }

    # 2) 有效月不足：soft=False → 维持原报错；soft=True → 软退化
    if n_valid < min_valid:
        if not soft:
            raise ValueError("有效月份少于 8，无法稳定拟合。")
        if n_valid < 4:
            return _empty_result()  # 月份太少，直接返回 NaN 参数但不中断流程

    # 3) 最小二乘
    m = np.arange(1, 13, dtype=float)
    X = np.column_stack([np.ones_like(m), np.cos(2*np.pi*m/12.0), np.sin(2*np.pi*m/12.0)])
    Xv, yv = X[mask, :], y_values[mask]
    if Xv.shape[0] < 3:
        return _empty_result()

    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    c, a, b = float(beta[0]), float(beta[1]), float(beta[2])

    amplitude = float(np.hypot(a, b))
    phase_rad = float(np.arctan2(b, a))
    phase_deg = float(np.degrees(phase_rad))
    if not np.isfinite(amplitude) or amplitude < 1e-6:
        phase_rad = np.nan
        phase_deg = np.nan
    
    if np.isfinite(phase_deg):
        peak_month = int((phase_deg % 360.0) / 30.0)
        peak_month = 12 if peak_month == 0 else peak_month
    else:
        peak_month = np.nan
    trough_month = ((peak_month + 5 - 1) % 12 + 1) if np.isfinite(peak_month) else np.nan
    trough_month = ((peak_month + 5 - 1) % 12) + 1

    recon_vals = c + a*np.cos(2*np.pi*m/12.0) + b*np.sin(2*np.pi*m/12.0)
    recon = xr.DataArray(recon_vals, coords={"month": np.arange(1,13)}, dims=("month",))
    recon.attrs.update(y.attrs)
    recon.attrs["long_name"] = (recon.attrs.get("long_name", "") + " first-harmonic fit").strip()

    ss_res = float(np.nansum((y_values[mask] - recon_vals[mask])**2))
    ss_tot = float(np.nansum((y_values[mask] - np.nanmean(y_values[mask]))**2))
    r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else np.nan

    return {
        "a": a, "b": b, "c": c,
        "amplitude": amplitude,
        "phase_rad": phase_rad, "phase_deg": phase_deg,
        "peak_month": int(peak_month), "trough_month": int(trough_month),
        "r2": r2,
        "recon": recon, "climatology": y,
        "n_valid": n_valid, "low_valid": (n_valid < min_valid),
    }

