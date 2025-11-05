#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate harmonics (A, phi) from output/harmonics/ into one tall table:
  output/tables/Table1_harmonics_A_phi.csv

It ingests:
  - cllmodis_harmonics_*.csv
  - clswlow_harmonics_*.csv, cllwlow_harmonics_*.csv
  - ERA5_*_harmonics_*.csv
  - CERES_*_harmonics_*.csv   (if present in future)

Expected columns (robust to presence/absence of extras):
  ["source","var","region","span","amplitude","phase_deg","R2","n_valid"]
"""

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

# Robust imports
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

try:
    from scripts.utils_config import load_config, make_output_path, setup_logger
except ModuleNotFoundError:
    from utils_config import load_config, make_output_path, setup_logger


def _read_one_csv(p: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"[WARN] cannot read {p.name}: {e}")
        return None

    # 标准化列名（兼容不同脚本）
    cols = {c.lower(): c for c in df.columns}
    # 推断 source/var/region/span
    source = None
    var = None
    region = None
    span = None

    # 1) 显式列
    for k in ["source","var","region","span"]:
        if k in cols:
            locals()[k] = df[cols[k]].iloc[0] if k != "region" else None  # region 每行可能不同，下面再处理

    # 2) 从文件名推断（兜底）
    name = p.name.lower()

    if source is None:
        if name.startswith("era5_"): source = "ERA5"
        elif name.startswith("ceres_"): source = "CERES"
        elif name.startswith("cllmodis"): source = "MODIS"
        elif name.startswith("clswlow"): source = "LCRE_LOW"
        elif name.startswith("cllwlow"): source = "LCRE_LOW"
        else: source = "OBS"

    if var is None:
        if name.startswith("era5_"):
            var = name.split("_", 2)[1].upper()  # ERA5_EIS_harmonics_...
        elif name.startswith("ceres_"):
            var = name.split("_", 2)[1].upper()
        elif name.startswith("cllmodis"):
            var = "LCF"
        elif name.startswith("clswlow"):
            var = "SWLCRE_low"
        elif name.startswith("cllwlow"):
            var = "LWLCRE_low"
        else:
            var = "UNK"

    # region：列里一般有
    if region is None:
        # 尝试列
        if "region" in cols:
            region_series = df[cols["region"]]
        else:
            # 从文件名最后一个下划线段取 region（弱推断）
            # e.g., ..._NEP.csv
            stem = p.stem
            token = stem.split("_")[-1]
            region_series = pd.Series([token]*len(df))
        df["region_norm"] = region_series
    else:
        df["region_norm"] = region

    # span：列里可能有
    if span is None and "span" in cols:
        span = df[cols["span"]].iloc[0]

    # 数值列名兼容
    amp_col = "amplitude" if "amplitude" in df.columns else cols.get("amplitude","amplitude")
    phi_col = "phase_deg" if "phase_deg" in df.columns else cols.get("phase_deg","phase_deg")
    R2_col  = "r2" if "r2" in df.columns else cols.get("r2","r2")
    nvalid_col = "n_valid" if "n_valid" in df.columns else cols.get("n_valid","n_valid")

    # 构造标准化表
    out = pd.DataFrame({
        "source": source,
        "var": var,
        "region": df["region_norm"],
        "span": span if span is not None else "",
        "A": df[amp_col] if amp_col in df.columns else pd.NA,
        "phi_deg": df[phi_col] if phi_col in df.columns else pd.NA,
        "R2": df[R2_col] if R2_col in df.columns else pd.NA,
        "n_valid": df[nvalid_col] if nvalid_col in df.columns else pd.NA,
        "file": p.name,
    })
    return out


def main():
    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    logger = setup_logger(cfg, name="make_table1_harmonics")

    harm_dir = Path(cfg["output"]["root"]) / cfg["output"]["subdirs"]["harmonics"]
    if not harm_dir.exists():
        raise SystemExit(f"harmonics dir not found: {harm_dir}")

    # 按常见前缀收集
    patterns = [
        "cllmodis_harmonics_*.csv",
        "clswlow_harmonics_*.csv",
        "cllwlow_harmonics_*.csv",
        "ERA5_*_harmonics_*.csv",
        "CERES_*_harmonics_*.csv",
    ]
    files = []
    for pat in patterns:
        files += list(harm_dir.glob(pat))

    if not files:
        raise SystemExit(f"No harmonics CSV found under {harm_dir}")

    rows = []
    for p in sorted(files):
        df1 = _read_one_csv(p)
        if df1 is not None:
            rows.append(df1)

    if not rows:
        raise SystemExit("No valid harmonics rows aggregated.")

    out_df = pd.concat(rows, ignore_index=True)

    # 排序：source -> var -> region
    order = ["source","var","region","span","A","phi_deg","R2","n_valid","file"]
    out_df = out_df[order].sort_values(["source","var","region"]).reset_index(drop=True)

    out_csv = make_output_path(cfg, "tables", "Table1_harmonics_A_phi.csv")
    out_df.to_csv(out_csv, index=False)
    logger.info(f"Saved Table 1 → {out_csv}")
    print(f"Saved Table 1 → {out_csv}")


if __name__ == "__main__":
    main()
