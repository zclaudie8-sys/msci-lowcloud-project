#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obs_periodogram_test.py

1) 读取 output/regional_monthly/cllmodis_mean_2003-2022_<REGION>.csv （需含列: month,value）
2) 计算功率谱：
   - 若月份等间隔且齐全（1..12），用 scipy.signal.periodogram (fs=1 month^-1)
   - 否则自动用 Lomb–Scargle (scipy.signal.lombscargle)
3) 拟合 AR(1) 系数 phi（基于 value 去均值后的 lag-1 自相关），构建红噪声谱：
   P_red(f) = sigma2 * (1 - phi^2) / (1 + phi^2 - 2 phi cos(2π f))
4) 用 2 自由度卡方近似给出 95% 阈值：P95(f) = P_red * chi2.ppf(0.95, 2) / 2
   并在 f0 = 1/12 month^-1 处判断功率是否显著（p-value 也会给出）。
5) 输出：
   - 图：fig/periodogram_<REGION>.png （观测功率谱 + 红噪声谱 + 95% 阈值 + 年频率标记）
   - 表：obs/periodogram_<REGION>.csv （列：freq, P_obs, P_red, p_value）

依赖：pandas, numpy, scipy, matplotlib
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats


# ---------------------------- utils ----------------------------

def log(msg: str) -> None:
    print(f"[periodogram] {msg}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Observed periodogram vs AR(1) red-noise test at f=1/12.")
    p.add_argument("--region", required=True, help="Region key, e.g. NEP/NEA/SEP/SEA/SEI")
    p.add_argument("--infile_tpl", default="output/regional_monthly/cllmodis_region_mean_2003-2022_{region}.csv",
                   help="Input CSV模板（需列 month,value）")
    p.add_argument("--outdir_fig", default="fig", help="图输出目录")
    p.add_argument("--outdir_tab", default="obs", help="表输出目录")
    p.add_argument("--use_logy", action="store_true", help="y轴用log尺度显示功率")
    return p.parse_args()

def ensure_dirs(*ps: str | Path) -> None:
    for p in ps:
        Path(p).mkdir(parents=True, exist_ok=True)

def fit_ar1_phi(x: np.ndarray) -> float:
    """基于去均值序列的 lag-1 自相关估计 AR(1) phi（若方差为0或不足样本，返回0）。"""
    x = np.asarray(x, float)
    x = x - np.nanmean(x)
    if np.isnan(x).any() or len(x) < 3:
        return 0.0
    x0 = x[:-1]; x1 = x[1:]
    denom = np.sum((x0 - x0.mean())**2)
    if denom <= 0:
        return 0.0
    phi = np.sum((x0 - x0.mean()) * (x1 - x1.mean())) / denom
    # 限幅防数值不稳
    return float(np.clip(phi, -0.99, 0.99))

def red_noise_spectrum(sigma2: float, phi: float, freq: np.ndarray) -> np.ndarray:
    """AR(1) 红噪声谱（Δt=1月）：P(f) = σ^2 (1-φ^2) / (1 + φ^2 - 2φ cos 2πf)"""
    w = 2 * np.pi * freq
    denom = 1 + phi**2 - 2 * phi * np.cos(w)
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    return sigma2 * (1 - phi**2) / denom

def chi2_p_value(P_obs: np.ndarray, P_red: np.ndarray, dof: int = 2) -> np.ndarray:
    """
    在期望功率 P_red 下，Periodogram 点的 (2*P_obs / P_red) ~ χ²_dof。
    返回右尾 p 值：p = 1 - CDF( (2*P_obs)/P_red ).
    """
    ratio = 2.0 * np.asarray(P_obs) / np.asarray(P_red)
    return 1.0 - stats.chi2.cdf(ratio, dof)


# ---------------------------- core ----------------------------

def compute_periodogram(months: np.ndarray, values: np.ndarray):
    """
    计算功率谱（Periodogram 或 Lomb-Scargle）。
    返回 freq, P_obs
    - freq 单位：cycles per month（month^-1）
    """
    # 清理
    m = np.asarray(months, float)
    y = np.asarray(values, float)
    ok = np.isfinite(m) & np.isfinite(y)
    m = m[ok]; y = y[ok]
    y = y - np.mean(y)

    # 等间隔判定：月差都≈1？
    dif = np.diff(np.sort(m))
    even = (len(m) >= 8) and np.allclose(dif, np.round(dif), atol=1e-6) and np.all(np.round(dif) == 1)

    if even and len(np.unique(m)) == len(m):
        # 等间隔 periodogram
        fs = 1.0  # 1 / month
        f, Pxx = signal.periodogram(y, fs=fs, window="hann", detrend="linear", scaling="density", return_onesided=True)
        # 去除 f=0（均值）
        keep = f > 0
        return f[keep], Pxx[keep], "periodogram"
    else:
        # Lomb–Scargle（scipy.signal.lombscargle 需角频率）
        # 构造频率轴：从 1/N 到 0.5（Nyquist）之间
        N = len(m)
        f = np.linspace(1.0/max(12, N), 0.5, 256)  # 更平滑的频率网格
        w = 2 * np.pi * f
        # 这里的实现是原始 Lomb–Scargle, 需要中心化
        y0 = y - np.mean(y)
        # scipy.signal.lombscargle 需要 x 单位一致；我们用“月”为单位，频率用 cycles/month -> 转成弧度频率即可
        # 注意：输入 x 是时间点（这里直接用月份序列），必须升序
        order = np.argsort(m)
        wP = signal.lombscargle(m[order], y0[order], w)
        # 标准化（近似）：P ~ (2/N)*wP，使量纲与 periodogram 接近
        P = (2.0 / N) * wP
        return f, P, "lombscargle"


# ---------------------------- plotting ----------------------------

def make_figure(freq, P_obs, P_red, P95, region: str, out_png: Path, use_logy: bool, f0: float, sig: bool):
    plt.figure(figsize=(8.4, 5.6))
    ax = plt.gca()

    # 观测谱
    ax.plot(freq, P_obs, lw=1.8, color="#1f77b4", label="Observed spectrum")
    # 红噪声谱
    ax.plot(freq, P_red, lw=1.4, color="#d62728", alpha=0.9, label="AR(1) red-noise")
    # 95% 阈值
    ax.plot(freq, P95, lw=1.2, color="#2ca02c", ls="--", alpha=0.9, label="95% threshold")

    # 年频率标记
    ax.axvline(f0, color="k", lw=1.2, ls="-.", alpha=0.9)
    ax.text(f0, ax.get_ylim()[1]*0.92, "annual (1/12)", ha="center", va="top", fontsize=9, rotation=90)

    ax.set_xlabel("Frequency (cycles per month)")
    ax.set_ylabel("Power")
    if use_logy:
        ax.set_yscale("log")

    ttl = f"Periodogram — {region} | annual peak {'significant' if sig else 'not significant'} at 95%"
    ax.set_title(ttl, fontsize=12)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------- main ----------------------------

# ---------------------------- main ----------------------------

import re
def detect_region_from_name(path: Path) -> str:
    # 例如 cllmodis_region_mean_2003-2022_NEP.csv 取 NEP
    m = re.search(r"_([A-Z]{3})\.csv$", path.name)
    return m.group(1) if m else "REG"

def read_and_make_climatology(infile: Path):
    """
    输入既可为:
      A) 气候态:  至少包含 month, value
      B) 时间序列: 至少包含 time, <var>  (如 cllmodis)
         -> 自动按 calendar month 聚合成年循环(12×1)
    返回: months(12,), values(12,)
    """
    df = pd.read_csv(infile)
    cols = {c.lower(): c for c in df.columns}

    # Case A: 已有 month,value
    if ("month" in cols) and ("value" in cols):
        ser = df[[cols["month"], cols["value"]]].dropna().copy()
        ser.columns = ["month", "value"]
        # 确保月份为 1..12
        if not np.issubdtype(ser["month"].dtype, np.number):
            ser["month"] = pd.to_numeric(ser["month"], errors="coerce")
        ser = ser.dropna().sort_values("month")
        return ser["month"].values.astype(float), ser["value"].values.astype(float)

    # Case B: time + 变量
    # time 列名
    tcol = cols.get("time") or cols.get("date") or cols.get("datetime")
    if tcol is None:
        raise ValueError(f"输入既无 (month,value) 也无 time 列：{infile}")
    # value 列名：优先 value，否则除 time 外取最后一列
    vcol = cols.get("value")
    if vcol is None:
        non_time_cols = [c for c in df.columns if c != tcol]
        if not non_time_cols:
            raise ValueError(f"未找到数值列：{infile}")
        vcol = non_time_cols[-1]

    # 解析日期 -> 月份
    ts = df[[tcol, vcol]].dropna().copy()
    ts.columns = ["time", "value"]
    ts["time"] = pd.to_datetime(ts["time"], errors="coerce")
    ts = ts.dropna(subset=["time"])
    ts["month"] = ts["time"].dt.month

    # 多年平均 -> 12×1
    clim = ts.groupby("month", as_index=False)["value"].mean().sort_values("month")
    return clim["month"].values.astype(float), clim["value"].values.astype(float)

'''
def run_one_region(infile: Path, region_name: str, out_dir_fig: Path, out_dir_tab: Path, use_logy: bool):
    log(f"读取：{infile}")
    months, values = read_and_make_climatology(infile)
    # 计算功率谱
    freq, P_obs, method = compute_periodogram(months, values)
    log(f"[{region_name}] 谱方法：{method}；点数={len(freq)}")

    # 红噪声
    x = values - np.mean(values)
    phi = fit_ar1_phi(x)
    sigma2 = float(np.var(x, ddof=1))
    P_red = red_noise_spectrum(sigma2, phi, freq)
    P95 = P_red * stats.chi2.ppf(0.95, df=2) / 2.0

    # 年频显著性
    f0 = 1.0 / 12.0
    i0 = int(np.argmin(np.abs(freq - f0)))
    p0 = float(chi2_p_value(P_obs[i0], P_red[i0], dof=2))
    sig = p0 < 0.05
    log(f"[{region_name}] f=1/12: p={p0:.3f}  → {'显著' if sig else '不显著'}")

    # 表
    out_csv = out_dir_tab / f"periodogram_{region_name}.csv"
    pvals = chi2_p_value(P_obs, P_red, dof=2)
    pd.DataFrame({"freq":freq, "P_obs":P_obs, "P_red":P_red, "p_value":pvals}).to_csv(out_csv, index=False)

    # 图
    out_png = out_dir_fig / f"periodogram_{region_name}.png"
    make_figure(freq, P_obs, P_red, P95, region_name, out_png, use_logy, f0, sig)
    log(f"[{region_name}] DONE → {out_png}")

def main():
    args = parse_args()
    out_dir_fig = Path(args.outdir_fig)
    out_dir_tab = Path(args.outdir_tab)
    ensure_dirs(out_dir_fig, out_dir_tab)

    # 支持 ALL 或逗号分隔
    regions_arg = args.region.upper()
    if regions_arg == "ALL":
        # 用模板搜全部文件：把 {region} 替换为 * ，再遍历
        pattern = args.infile_tpl.format(region="*")
        files = sorted(map(Path, list(Path().glob(pattern))))
        if not files:
            raise FileNotFoundError(f"ALL 模式未找到任何文件：{pattern}")
        for f in files:
            rname = detect_region_from_name(f)
            run_one_region(f, rname, out_dir_fig, out_dir_tab, args.use_logy)
    elif "," in regions_arg:
        for r in [s.strip().upper() for s in regions_arg.split(",") if s.strip()]:
            infile = Path(args.infile_tpl.format(region=r))
            if not infile.exists():
                log(f"[WARN] 文件不存在，跳过：{infile}")
                continue
            run_one_region(infile, r, out_dir_fig, out_dir_tab, args.use_logy)
    else:
        r = regions_arg
        infile = Path(args.infile_tpl.format(region=r))
        if not infile.exists():
            raise FileNotFoundError(f"未找到输入文件：{infile}")
        run_one_region(infile, r, out_dir_fig, out_dir_tab, args.use_logy)
'''
def run_one_region(infile: Path, region_name: str, out_dir: Path, use_logy: bool):
    log(f"读取：{infile}")
    months, values = read_and_make_climatology(infile)
    freq, P_obs, method = compute_periodogram(months, values)
    log(f"[{region_name}] 谱方法：{method}；点数={len(freq)}")

    # --- 红噪声谱 ---
    x = values - np.mean(values)
    phi = fit_ar1_phi(x)
    sigma2 = float(np.var(x, ddof=1))
    P_red = red_noise_spectrum(sigma2, phi, freq)
    P95 = P_red * stats.chi2.ppf(0.95, df=2) / 2.0

    # --- 年频显著性 ---
    f0 = 1.0 / 12.0
    i0 = int(np.argmin(np.abs(freq - f0)))
    p0 = float(chi2_p_value(P_obs[i0], P_red[i0], dof=2))
    sig = p0 < 0.05
    log(f"[{region_name}] f=1/12: p={p0:.3f} → {'显著' if sig else '不显著'}")

    # ✅ 输出路径统一为 output/rednoisefile/
    out_csv = out_dir / f"rednoisetest_{region_name}.csv"
    out_png = out_dir / f"rednoisetest_{region_name}.png"

    # --- 写入 CSV ---
    pvals = chi2_p_value(P_obs, P_red, dof=2)
    pd.DataFrame({
        "freq": freq,
        "P_obs": P_obs,
        "P_red": P_red,
        "p_value": pvals
    }).to_csv(out_csv, index=False)

    # --- 绘制图像 ---
    make_figure(freq, P_obs, P_red, P95, region_name, out_png, use_logy, f0, sig)
    log(f"[{region_name}] DONE → {out_png}")


def main():
    args = parse_args()
    # ✅ 固定输出目录
    out_dir = Path("output/rednoisefile")
    out_dir.mkdir(parents=True, exist_ok=True)

    regions_arg = args.region.upper()
    if regions_arg == "ALL":
        pattern = args.infile_tpl.format(region="*")
        files = sorted(map(Path, list(Path().glob(pattern))))
        if not files:
            raise FileNotFoundError(f"ALL 模式未找到任何文件：{pattern}")
        for f in files:
            rname = detect_region_from_name(f)
            run_one_region(f, rname, out_dir, args.use_logy)
    elif "," in regions_arg:
        for r in [s.strip().upper() for s in regions_arg.split(",") if s.strip()]:
            infile = Path(args.infile_tpl.format(region=r))
            if not infile.exists():
                log(f"[WARN] 文件不存在，跳过：{infile}")
                continue
            run_one_region(infile, r, out_dir, args.use_logy)
    else:
        r = regions_arg
        infile = Path(args.infile_tpl.format(region=r))
        if not infile.exists():
            raise FileNotFoundError(f"未找到输入文件：{infile}")
        run_one_region(infile, r, out_dir, args.use_logy)



if __name__ == "__main__":
    main()
