#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_cmip_panel.py — 生成 CMIP 月度面板（time, model, region, value），供 feedback_fit 使用。
默认 2003–2014；按 configs/config.yaml 的区域；只取 AMIP 文件（*_amip_*.nc）。

用法示例：
  python scripts/make_cmip_panel.py --var cllisccp
  python scripts/make_cmip_panel.py --var ts
  python scripts/make_cmip_panel.py --var swcre
  python scripts/make_cmip_panel.py --var eislts

  python scripts/make_cmip_panel.py --var swcre --span 2003-2014   # 会自动去 clswlow/
python scripts/make_cmip_panel.py --var lcf   --span 2003-2014   # 自动去 cllisccp/
python scripts/make_cmip_panel.py --var eis   --span 2003-2014   # 自动去 eislts/（或按你修改）

"""
from __future__ import annotations
from pathlib import Path
import sys, argparse, logging
import xarray as xr
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))

try:
    from scripts.utils_config import load_config, get_regions, make_output_path, setup_logger
    from scripts.utils_region  import select_region_mean
except ModuleNotFoundError:
    from utils_config import load_config, get_regions, make_output_path, setup_logger
    from utils_region  import select_region_mean

CMIP_ROOT = Path("/gws/nopw/j04/csgap/ceppi/data/cmip6/5x5")

VAR_DIR_ALIASES = {
    "swcre":  "clswlow",   # 短波低云CRE
    "lcf":    "cllisccp",
    "eis":    "eislts",    # 若你只有 eis，可把值改成 "eis"
    "tas":    "tas",       # 2m 气温（若你用 tas 做近地表温度）
    "ts":     "ts",        # 地表温度
}
def parse_args():
    p = argparse.ArgumentParser(description="Build CMIP AMIP monthly panel CSV for a variable.")
    p.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    p.add_argument("--var", required=True, help="变量目录名（如 cllisccp / ts / swcre / eislts）")
    p.add_argument("--span", default="2003-2014", help="年段，形如 2003-2014")
    return p.parse_args()

def _parse_span(span: str) -> tuple[int,int]:
    a,b = span.split("-"); y0,y1 = int(a), int(b)
    return (y0,y1) if y0<=y1 else (y1,y0)
def _safe_datetime_index(time_coord, y0, y1):
    try:
        idx = time_coord.to_index()
        if isinstance(idx, pd.DatetimeIndex):
            return idx
    except Exception:
        pass
    try:
        idx = time_coord.to_index().to_datetimeindex()
        return idx
    except Exception:
        pass
    n = time_coord.sizes.get("time", len(time_coord))
    return pd.date_range(f"{y0}-01-01", periods=n, freq="MS")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger(cfg, name=f"make_cmip_panel_{args.var}")
    y0,y1 = _parse_span(args.span)

    var = args.var.strip()
    var_dirname = VAR_DIR_ALIASES.get(var, var)   # 有别名就用别名目录，否则用原值
    var_dir = CMIP_ROOT / var_dirname

    if not var_dir.exists():
        raise SystemExit(f"[ERR] 目录不存在：{var_dir}")

    regdict = get_regions(cfg)  # {'NEP':[latmin,latmax,lonmin,lonmax], ...}
    files = sorted(var_dir.glob("*_amip_*.nc"))
    if not files:
        raise SystemExit(f"[ERR] 未找到 AMIP 文件：{var_dir}/**/*_amip_*.nc")

    rows = []
    for f in files:
        model = f.name.split("_")[0]  # 约定：<model>_*_amip_*.nc
        try:
            ds = xr.open_dataset(f, decode_times=True)
        except Exception as e:
            logger.error(f"[OPEN FAIL] {f.name}: {e}")
            continue

        if var not in ds and var.upper() in ds:
            da = ds[var.upper()]
        else:
            da = ds.get(var)
        if da is None:
            logger.error(f"[VAR MISSING] {f.name} 无变量 '{var}'；可用：{list(ds.data_vars)}")
            continue

        # 年段切片
        da = da.where((da["time"].dt.year >= y0) & (da["time"].dt.year <= y1), drop=True)
        if da.time.size == 0:
            logger.warning(f"[EMPTY SPAN] {f.name} 在 {y0}-{y1} 无数据")
            continue

        # 每个区域做面积均值
        for region, box in regdict.items():
            # utils_region 里区域是 [lat_min, lat_max, lon_min, lon_max]
            try:
                ts = select_region_mean(da, [box[2], box[3], box[0], box[1]])  # W,E,S,N
            except Exception:
                # 某些 select_region_mean 已支持 [lat_min, lat_max, lon_min, lon_max]，再试一次
                ts = select_region_mean(da, box)
            idx = _safe_datetime_index(ts["time"], y0, y1)
            s   = pd.Series(ts.values.astype(float), index=idx)
            s = s.sort_index()
            rows.append(pd.DataFrame({
                "time": s.index,
                "model": model,
                "region": region,
                "value": s.values.astype(float)
            }))

        ds.close()

    if not rows:
        raise SystemExit("[ERR] 没有生成任何面板数据。请检查变量名、年段与文件。")

    panel = pd.concat(rows, ignore_index=True).sort_values(["model","region","time"])
    out = make_output_path(cfg, "tables", f"cmip_panel_{var}_{y0}-{y1}.csv")
    panel.to_csv(out, index=False)
    logger.info(f"[OUT] {out}  (cols: time, model, region, value)")

if __name__ == "__main__":
    main()
