#!/usr/bin/env python3
# Rebuild CMIP wide table with clean keys and inner-join intersection
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("output/tables")  # 如路径不同改这里
OUT  = Path("output/cmip_amip_monthly_2003-2014_wide.csv")

# ---- 可选参数：如何处理多 member/variant ----
SELECT_MEMBER = True          # True: 选择单个 member；False: 对 member 求均值
PREFERRED_MEMBER_PREFIXES = ("r1i1", "r1i1p1", "r1i1p1f1")  # 优先选这些
# -------------------------------------------

def read_panel(path: Path, target_name: str, candidates: tuple[str,...]) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[ERR] missing file: {path}")
    df = pd.read_csv(path, low_memory=False)
    cols = {c.lower(): c for c in df.columns}

    # 1) 找数值列
    vcol = None
    for k in (target_name.lower(), "value", "val", "y", "x", *candidates):
        if k.lower() in cols:
            vcol = cols[k.lower()]
            break
    if vcol is None:
        sys.exit(f"[ERR] {path.name}: cannot find value column for {target_name}; "
                 f"available: {list(df.columns)}")

    # 2) 找时间列
    tcol = None
    for k in ("time","date","month"):
        if k in cols:
            tcol = cols[k]; break
    if tcol is None:
        sys.exit(f"[ERR] {path.name}: cannot find time column (time/date/month)")

    # 3) 标准列
    out = df.rename(columns={vcol: target_name, tcol: "time"}).copy()
    # 保留可能的 member/variant 信息
    meta_cols = []
    for k in ("member","variant","variant_label","realization","ensemble"):
        if k in cols:
            meta_cols.append(cols[k])

    need = ["model","region","time",target_name] + meta_cols
    miss = [c for c in ["model","region","time"] if c not in out.columns]
    if miss:
        sys.exit(f"[ERR] {path.name}: missing essential key columns {miss}")

    # 4) 解析时间并“月对齐”
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["time"])
    # 月度统一到月初（避免 'YYYY-MM' vs 'YYYY-MM-01' 混合）
    out["time"] = out["time"].dt.to_period("M").dt.to_timestamp("M")  # 月末
    # 如你习惯用月初可改为 .dt.to_period("M").dt.to_timestamp()

    # 5) 只保留必要列
    out = out[["model","region","time",target_name] + meta_cols]

    # 6) 去重（完全重复）
    out = out.drop_duplicates()

    # 7) 处理 member：选择或聚合
    key = ["model","region","time"]
    if meta_cols and SELECT_MEMBER:
        # 优先选常见的 r1i1...；否则选出现次数最多的那个
        def pick_member(g: pd.DataFrame) -> pd.DataFrame:
            g2 = g.copy()
            # 组装一个 member_label 字段
            for mcol in meta_cols:
                if mcol in g2.columns:
                    label = g2[mcol].astype(str)
                    break
            else:
                label = pd.Series([""]*len(g2))
            g2 = g2.assign(_member_label=label)
            # 找优先 member
            order = np.argsort(g2["_member_label"].apply(lambda s: (
                0 if any(str(s).startswith(p) for p in PREFERRED_MEMBER_PREFIXES) else 1, str(s)
            )).to_list())
            # 取排序后的第一行
            chosen = g2.iloc[order[:1]]
            return chosen[key + [target_name]]
        out = (out.groupby(key, as_index=False, sort=False)
                    .apply(pick_member)
                    .reset_index(drop=True))
    elif meta_cols and not SELECT_MEMBER:
        # 对 member 取均值
        out = (out.groupby(key, as_index=False)[target_name].mean())
    else:
        # 没有 member 列，若同一 key 仍有重复，则取均值
        out = (out.groupby(key, as_index=False)[target_name].mean())

    return out

def main():
    sw_path  = BASE/"cmip_panel_clswlow_2003-2014_models.csv"
    eis_path = BASE/"cmip_panel_eislts_2003-2014_models.csv"
    ts_path  = BASE/"cmip_panel_ts_2003-2014_models.csv"

    SWCRE = read_panel(sw_path,  "SWCRE", ("clswlow","swcre","swcre_mean","sw_lcre","swcre_low"))
    EIS   = read_panel(eis_path, "EIS",   ("eislts","eis"))
    Ts    = read_panel(ts_path,  "Ts",    ("tas","ts","surface_temperature"))

    print("[INFO] rows SWCRE/EIS/Ts:", len(SWCRE), len(EIS), len(Ts))

    # 统计每个表中每组(模型×区域)含多少月
    def count_months(df, name):
        g = df.groupby(["model","region"]).size().reset_index(name=f"n_{name}")
        print(f"[INFO] per (model,region) counts in {name}, head:")
        print(g.sort_values(f"n_{name}").head(8))
        return g
    count_months(SWCRE, "SWCRE"); count_months(EIS, "EIS"); count_months(Ts, "Ts")

    # INNER JOIN：只保留三者共有的 (model, region, time)
    wide = (SWCRE.merge(EIS, on=["model","region","time"], how="inner")
                  .merge(Ts,  on=["model","region","time"], how="inner"))

    # 再去一次重
    wide = wide.drop_duplicates()
    # 简单质量控制：全非空
    wide = wide.dropna(subset=["SWCRE","EIS","Ts"])

    print("[INFO] wide rows:", len(wide))
    print("[INFO] wide head:")
    print(wide.head(6))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT, index=False)
    print("[INFO] WROTE:", OUT, "columns:", list(wide.columns))

    # 快速核查：每个 (region,model) 的完整月数
    def full_rows(g): return g.shape[0]
    chk = (wide.groupby(["region","model"]).apply(full_rows)
                 .reset_index(name="n_full")
                 .sort_values("n_full"))
    print("[INFO] n_full head (should be near 144):")
    print(chk.head(15))

if __name__ == "__main__":
    main()
