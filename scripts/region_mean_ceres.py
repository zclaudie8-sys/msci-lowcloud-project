#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CERES regional monthly means (STRICT, NO GUESSING)

做什么：
  - 对 CERES 变量（SWCRE/LWCRE/NET/LCRE 等）做区域面积加权月均
  - 严格按 config.yaml 中 data.ceres_vars 的 {file, var} 映射读取（不猜文件名/变量名）

配置示例（请在 configs/config.yaml 中添加/修改）：
data:
  ceres_path: /gws/nopw/j04/csgap/ceppi/data/ceres/5x5/
  ceres_vars:
    SWCRE: {file: "EBAF_TOA_5x5_200003-202412.nc", var: "swcre"}
    LWCRE: {file: "EBAF_TOA_5x5_200003-202412.nc", var: "lwcre"}
    NET:   {file: "EBAF_TOA_5x5_200003-202412.nc", var: "net"}      # 若无，可先留空或去掉
    LCRE:  {file: "LCRE_low_5x5_200207-202406.nc",  var: "lcre"}    # 仅当你有“低云 CRE”产品

用法：
  # 列出目录下每个 nc 的变量清单（不计算）
  python scripts/region_mean_ceres.py --list

  # 校验映射是否正确（不计算）
  python scripts/region_mean_ceres.py --check

  # 计算（区域月均），示例：
  python scripts/region_mean_ceres.py --region NEP --vars SWCRE,LWCRE,NET,LCRE
  # 计算 all region
   python scripts/region_mean_ceres.py --vars SWCRE,LWCRE,LCRE     --regions ALL
"""

from __future__ import annotations
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import xarray as xr

# 兼容从项目根或 scripts/ 运行
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

try:
    from scripts.utils_config import load_config, get_regions, make_output_path, setup_logger
    from scripts.utils_region  import select_region_mean
except ModuleNotFoundError:
    from utils_config import load_config, get_regions, make_output_path, setup_logger
    from utils_region import select_region_mean


def parse_args():
    p = argparse.ArgumentParser(description="CERES regional monthly means (STRICT).")
    p.add_argument("--region", help="Region key in config (NEP/NEA/SEP/SEA/SEI)")
    p.add_argument("--regions", default="", help="Comma list or ALL (use project.regions)")
    p.add_argument("--vars", help="Comma-separated logical names matching data.ceres_vars (e.g., SWCRE,LWCRE,NET)")
    p.add_argument("--list", action="store_true", help="List all nc files and their variables; no computation.")
    p.add_argument("--check", action="store_true", help="Validate config mappings; no computation.")
    return p.parse_args()

def _regions_to_run(cfg: dict, args) -> list[tuple[str, list[float]]]:
    if getattr(args, "regions", ""):
        rs = [r.strip() for r in args.regions.split(",") if r.strip()]
        if len(rs) == 1 and rs[0].upper() == "ALL":
            rs = cfg["project"]["regions"]
    elif getattr(args, "region", ""):
        rs = [args.region]
    else:
        raise SystemExit("请提供 --region 或 --regions（可用 ALL）")

    regdict = get_regions(cfg)
    out = []
    for r in rs:
        if r not in regdict:
            raise KeyError(f"未知区域键：{r}；可用={list(regdict.keys())}")
        raw_box = regdict[r]
        box = _cfg_latlat_lonlon_to_WESN(raw_box)
        out.append((r, box))
    return out


def _list_nc_vars(base: Path) -> list[dict]:
    rows = []
    for f in sorted(base.glob("*.nc")):
        try:
            with xr.open_dataset(f) as ds:
                rows.append({
                    "file": f.name,
                    "vars": list(ds.data_vars.keys()),
                    "dims": list(ds.dims.keys()),
                })
        except Exception as e:
            rows.append({"file": f.name, "vars": [f"<open failed: {e}>"], "dims": []})
    return rows


def _norm_list(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _cfg_latlat_lonlon_to_WESN(raw_box):
    """
    配置里的约定： [lat_min, lat_max, lon_min, lon_max]
    显式转成 [W, E, S, N]
    """
    if not (isinstance(raw_box, (list, tuple)) and len(raw_box) == 4):
        raise ValueError(f"区域必须是 [lat_min, lat_max, lon_min, lon_max]，给到的是：{raw_box}")
    lat_min, lat_max, lon_min, lon_max = map(float, raw_box)
    return [lon_min, lon_max, lat_min, lat_max]  # [W, E, S, N]


def _get_mapping(cfg: dict, key: str) -> tuple[Path, str]:
    m = cfg.get("data", {}).get("ceres_vars", {}).get(key)
    if not m:
        raise KeyError(f"[映射缺失] data.ceres_vars 中没有条目：{key}")
    f = m.get("file"); v = m.get("var")
    if not f or not v:
        raise KeyError(f"[映射不完整] {key} 需要 file 与 var：{m}")
    ceres_dir = Path(cfg["data"]["ceres_path"])
    fpath = ceres_dir / f
    if not fpath.exists():
        raise FileNotFoundError(f"[文件不存在] {key}: {fpath}")
    # 立即验证变量是否存在
    with xr.open_dataset(fpath) as ds:
        if v not in ds.data_vars:
            raise KeyError(
                f"[变量不存在] {key}: 期望 var='{v}' 不在文件 {fpath.name} 中。\n"
                f"可用变量：{list(ds.data_vars.keys())}"
            )
    return fpath, v



def main():
    args = parse_args()
    cfg = load_config(str(ROOT / "configs" / "config.yaml"))

    # --list
    if args.list:
        ceres_dir = Path(args.ceres_path or cfg["data"]["ceres_path"])
        print(f"列出目录：{Ceres_dir}")
        rows = _list_nc_vars(ceres_dir)
        if not rows:
            print(f"(空) 目录下没有 .nc：{ceres_dir}")
            return
        for r in rows:
            print(f"- {r['file']}: vars={r['vars']}")
        return

    # --check
    if args.check:
        ceres_dir = Path(args.ceres_path or cfg["data"]["ceres_path"])
        maps = cfg.get("data", {}).get("ceres_vars", {})
        if not maps:
            raise SystemExit("config.yaml 中 data.ceres_vars 为空。")
        ok = True
        for k, spec in maps.items():
            try:
                _get_mapping(cfg, k)  # 这里内部已使用 cfg 路径
                print(f"[OK] {k} -> {spec}")
            except Exception as e:
                ok = False
                print(f"[ERR] {k} -> {spec}\n  {e}")
        if not ok:
            raise SystemExit("有错误。请按上面提示修正 data.ceres_vars。")
        print("全部映射校验通过。")
        return

    # 计算模式
    if not args.vars:
        raise SystemExit("计算模式必须提供 --vars。")
    req_vars = [x.strip() for x in args.vars.split(",") if x.strip()]

    logger = setup_logger(cfg, name="region_mean_ceres")
    y0, y1 = cfg["project"]["time_span"]

    # === B. 决定要跑的区域列表 ===
    regions_to_run = _regions_to_run(cfg, args)

    # === C. 单区流水线 → 区域循环 ===
    for region_key, box in regions_to_run:
        logger.info(f"[RUN] region={region_key} | box={box} | span: {y0}-{y1}")
        series_list = []

        for key in req_vars:
            try:
                fpath, varname = _get_mapping(cfg, key)
            except Exception as e:
                logger.error(f"[{key}] 映射无效：{e}")
                raise

            logger.info(f"[{key}] using file={fpath.name} varname={varname}")
            try:
                ds = xr.open_dataset(fpath)
            except Exception as e:
                logger.exception(f"[{key}] 打开失败：{e}")
                continue

            da = ds[varname]
            ts = select_region_mean(da, box)
            ts = ts.where((ts["time"].dt.year >= y0) & (ts["time"].dt.year <= y1), drop=True)
            if ts.size == 0:
                logger.warning(f"[{key}] 在 {y0}-{y1} 无数据（或区域内全 NaN），跳过。")
                continue

            ser = ts.to_series(); ser.name = f"CERES_{key}"
            series_list.append(ser)

        # ---- 派生 NET（本区变量循环之后） ----
        try:
            _df_tmp = pd.concat(series_list, axis=1)
            if "CERES_SWCRE" in _df_tmp.columns and "CERES_LWCRE" in _df_tmp.columns:
                _df_tmp["CERES_NET"] = _df_tmp["CERES_SWCRE"] + _df_tmp["CERES_LWCRE"]
                series_list.append(_df_tmp["CERES_NET"])
                logger.info("Added derived column CERES_NET = CERES_SWCRE + CERES_LWCRE")
            else:
                logger.info("Skip CERES_NET: 需要同时存在 SWCRE 与 LWCRE")
        except Exception as e:
            logger.warning(f"自动生成 NET 失败（{e}）")

        if not series_list:
            logger.warning(f"[{region_key}] 本区没有成功产出的变量，跳过写月表。")
            continue

        df_monthly = pd.concat(series_list, axis=1).sort_index()
        out_csv = make_output_path(cfg, "regional_monthly", f"CERES_{region_key}_monthly.csv")
        df_monthly.to_csv(out_csv, index=True)
        logger.info(f"[{region_key}] Saved monthly table → {out_csv}")

    logger.info("DONE.")



if __name__ == "__main__":
    main()
