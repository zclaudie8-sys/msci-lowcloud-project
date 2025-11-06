# Output Map

## scripts/audit_outputs.py
Brief: Audit output paths used across analysis scripts.

- log | - Write execution logs to `logs/` with timestamps or identifiers. | write_docs | heuristic string match
- other | - Figures should be written to `fig/` or `figures/` folders with informative names. | write_docs | heuristic string match
- other | - Prefer storing tabular data in `tables/` with descriptive filenames. | write_docs | heuristic string match
- other | - Use `output/` for derived datasets and diagnostics. | write_docs | heuristic string match
- other | <DIRECTORY> | main | variable placeholder; directory creation
- other | <HANDLE> | write_json | variable placeholder
- other | output/ | <module> | heuristic string match

## scripts/augment_modis_optics.py
Brief: No module docstring detected.
Constants: ROOT = <HERE_PARENT>

- other | <HERE_PARENT> | <module> | constant definition; attribute placeholder
- table | <OUT> | main | variable placeholder

## scripts/dLCF_dTs_A_direct.py
Brief: A-line (direct) estimate of dLCF/dTs with meteorology controlled.
Constants: ROOT = <HERE_PARENT>

- log | 
A-line (direct) estimate of dLCF/dTs with meteorology controlled.

Model (per calendar month m):
    LCF'_m = theta_T^(m) * Ts'_m + sum_j theta_j^(m) * CCF'_{j,m} + eps_m

Inputs (per region):
  - output/regional_monthly/MODIS_<REGION>_monthly.csv   (need column MODIS_LCF; fallback to cllmodis_region_mean_*.csv)
  - output/regional_monthly/ERA5_<REGION>_monthly.csv    (ERA5_TS / ERA5_SST / ERA5_EIS / ERA5_W500 / ...)

Arguments:
  --region NEP --ccf EIS,W500,SST,U10,Q --lags 1

Outputs:
  - output/dLCF_dTs/ALINE_<REGION>.csv        (month, dLCF_dTs, se, t, p, ci_low, ci_high, n, rsq, Ts_name)
  - output/dLCF_dTs/ALINE_HARM_<REGION>.csv   (A, phase_deg, peak_month, r2, n_valid)
  - (append/update) output/tables/Table4_dLCF_dTs_A.csv (region, mean±CI, A, phase_deg)

Notes
- Per-month Z-score (by calendar month) for LCF, Ts and all CCFs.
- Ts predictor is chosen automatically:
    * If user lists SST in --ccf, Ts = ERA5_SST and SST is removed from CCFs to avoid duplication.
    * Else if ERA5_TS exists, Ts = ERA5_TS; else if ERA5_SST exists, Ts = ERA5_SST.
- Newey–West (HAC) SE with lag = --lags (default 1).
- VIF checked per month (info-level warning only).
 | <module> | heuristic string match
- other | <HERE_PARENT> | <module> | constant definition; attribute placeholder
- table | <OUT_CSV> | main | variable placeholder
- table | <OUT_HARM> | main | variable placeholder
- table | <TABLE4_PATH> | main | variable placeholder
- table | <TABLE4_PATH> | main | variable placeholder

## scripts/feedback_fit_cmip_vs_obs.py
Brief: Compute OBS vs CMIP6 AMIP feedback fits over 2003–2014.

- figure | <FIG_PATH> | make_scatter | variable placeholder
- log | <LOGS_DIR> | setup_logger | variable placeholder; directory creation
- other | <FIG_PATH_PARENT> | make_scatter | attribute placeholder; directory creation
- other | <TABLE_PATH_PARENT> | process_region | attribute placeholder; directory creation
- other | output/regional_monthly | process_region | heuristic string match
- table | <TABLE_PATH> | process_region | variable placeholder
- table | output/cmip_amip_monthly_2003-2014.csv | parse_args | heuristic string match

## scripts/feedback_fit_cmip_vs_obs_strict.py
Brief: feedback_fit_cmip_vs_obs_strict.py  (STRICT, config-aware, NO GUESSING)
Constants: ROOT = <HERE_PARENT>

- figure | <FIG_PATH> | plot_model_vs_obs | variable placeholder
- log | 
feedback_fit_cmip_vs_obs_strict.py  (STRICT, config-aware, NO GUESSING)

比较（默认 2003–2014，可用 --span 覆盖）：
  (1) λ_cld,SW = d(SWCRE)/d(Ts)
  (2) d(LCF)/d(EIS)

读取优先级（严格按你的目录）：
  1) CERES_<REGION>_monthly.csv   （SWCRE）
     ERA5_<REGION>_monthly.csv    （Ts / EIS）
     MODIS_<REGION>_monthly.csv   （LCF）
  2) 回退：{prefix}_region_mean_{span}_{REGION}.csv
          {prefix}_climatology_{span}_{REGION}.csv

统计：
  * y ~ β x + const
  * HAC(Newey–West) 标准误（--hac-lags；--no-hac 关闭）
  * AR(1) 报告：phi_y, phi_x, Ljung–Box(1) p 值（lb_p_y/x），Durbin–Watson（dw）

输出：
  * tables/feedback_fit_<y>_vs_<x>_<REGION>.csv
  * figures/feedback_fit_<y>_vs_<x>_<SCOPE>.png
  * logs/feedback_fit_*.log （由 setup_logger 写入）

  python scripts/feedback_fit_cmip_vs_obs_strict.py   --pair swcre_ts   --config configs/config.yaml   --regions ALL   --span 2003-2022   --deseasonalize --hac-lags 12   --obs-map SWCRE=swcre,Ts=ts   --cmip-y output/tables/cmip_panel_clswlow_2003-2014.csv   --cmip-x output/tables/cmip_panel_ts_2003-2014.csv
# 如果上一步生成的是 tas 面板，就把最后一行改成：
# --cmip-x output/tables/cmip_panel_tas_2003-2014.csv



 | <module> | heuristic string match
- other | <FIG_PATH_PARENT> | plot_model_vs_obs | attribute placeholder; directory creation
- other | <HERE_PARENT> | <module> | constant definition; attribute placeholder
- other | CMIP x 表（默认 output/cmip_amip_<x>_vs_obs_<span>.csv） | parse_args | heuristic string match
- other | CMIP y 表（默认 output/cmip_amip_<y>_vs_obs_<span>.csv） | parse_args | heuristic string match
- other | output/cmip_amip_ | main | heuristic string match
- table | <OUT_CSV> | main | variable placeholder
- table | <OUT_CSV_ALL> | main | variable placeholder

## scripts/gamma_radiative.py
Brief: Monthly radiative sensitivity:
Constants: ROOT = <HERE_PARENT>

- figure | <FIG_PATH> | main | variable placeholder
- other | 
    优先读 output/regional_monthly/MODIS_<REGION>_monthly.csv（需包含 MODIS_LCF；可选 MODIS_TAU / MODIS_RE）。
    若不存在，则回退到 cllmodis_region_mean_*_<REGION>.csv（两列或无表头），构造最小月表（仅 MODIS_LCF）。
     | _read_modis | heuristic string match
- other | 
Monthly radiative sensitivity:
  SWCRE'_m = gamma^(m) * LCF'_m + eta_m
Optionally binned by optical depth (TAU) or effective radius (RE):
  gamma(m | bin)

Inputs (per region):
  - output/regional_monthly/MODIS_<REGION>_monthly.csv   (MODIS_LCF; optional MODIS_TAU / MODIS_RE)
  - output/regional_monthly/CERES_<REGION>_monthly.csv   (CERES_SWCRE)

Preprocessing:
  - Align to month-start timestamps
  - Per-month Z-score for both SWCRE and LCF (and bin variable if needed)
  - OLS with Newey–West (HAC) SE (lag = --lags)

Outputs:
  - output/gamma/GAMMA_<REGION>.csv
  - output/gamma/GAMMA_HARM_<REGION>.csv
  - output/tables/Table3_gamma.csv   (append/update for this region; incl. mean±CI and A,phi)
  - output/figures/Fig3_swcre_vs_lcf_<REGION>.png

Usage:
  python scripts/gamma_radiative.py --region NEP --lags 1
  python scripts/gamma_radiative.py --region NEP --bin_by TAU --bins 3 --lags 1
  # 单区，不分箱
python scripts/gamma_radiative.py --region NEP --lags 1

# 单区，分箱（τ 三分位）
python scripts/gamma_radiative.py --region NEP --bin_by TAU --bins 3 --lags 1

# 一次跑五区（不分箱）
python scripts/gamma_radiative.py --regions ALL --lags 1

# 一次跑五区（分箱）
python scripts/gamma_radiative.py --regions ALL --bin_by RE --bins 4 --lags 1

 | <module> | heuristic string match
- other | <HERE_PARENT> | <module> | constant definition; attribute placeholder
- table | <OUT_CSV> | main | variable placeholder
- table | <OUT_HARM> | main | variable placeholder
- table | <TABLE3_PATH> | main | variable placeholder
- table | <TABLE3_PATH> | main | variable placeholder

## scripts/harmonic_summary_cmip_amip.py
Brief: CMIP6 AMIP harmonic summary vs observations (5x5 monthly)

- figure | <FIG1> | main | variable placeholder
- figure | <FIG1B> | main | variable placeholder
- figure | <FIG2> | main | variable placeholder
- figure | <FIG3> | main | variable placeholder
- other | 
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
 | <module> | heuristic string match
- table | <OUT_CSV> | main | variable placeholder

## scripts/harmonic_summary_obs.py
Brief: Summarize first-harmonic fitting for MODIS observations across regions.

- figure | <AMP_FIG> | main | variable placeholder
- figure | <PEAK_FIG> | main | variable placeholder
- other | 
Summarize first-harmonic fitting for MODIS observations across regions.

Outputs (under output/ and figures/):
  - output/<var>_harmonics_summary_<span>.csv
  - figures/<var>_amplitude_summary_<span>.png
  - figures/<var>_peakmonth_summary_<span>.png

Usage:
  python scripts/harmonic_summary_obs.py --var cllmodis
  # 可选变量: cllmodis, clswlow, cllwlow
 | <module> | heuristic string match
- table | <OUT_CSV> | main | variable placeholder

## scripts/harmonic_yearly_obs.py
Brief: harmonic_yearly_obs.py

- other | 
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
 | <module> | heuristic string match
- other | <HARMONICS_DIR> | main | variable placeholder; directory creation
- other | <ROSE_DIR> | main | variable placeholder; directory creation
- other | output/harmonics | main | heuristic string match
- table | <OUT_CSV> | main | variable placeholder

## scripts/lambda_chain.py
Brief: lambda_chain.py
Constants: ROOT = <HERE_PARENT>

- figure | <FIG_NAME> | main | variable placeholder
- other | 
lambda_chain.py

Compute monthly cloud feedback via product:
    lambda_cld^(m) = gamma^(m) * (dLCF/dTs)^(m)
and decompose month-to-month departures into gamma / dLCF_dTs / synergy.

Reads:
  - output/gamma/GAMMA_<REGION>__*.csv  (auto-pick binned_* or unbinned; use overall rows bin=NaN)
  - output/dLCF_dTs/ALINE_<REGION>.csv  (A-line; column dLCF_dTs)
  - output/dLCF_dTs/BCHAIN_<REGION>.csv (B-line; column dLCF_dTs_chain)

Args:
  --region NEP            # single region
  --regions ALL|SEP,SEA   # run multiple
  --method A|B|BOTH       # default BOTH

Outputs per region:
  - output/lambda/LAMBDA_<REGION>_A.csv, ..._B.csv
  - output/lambda/LAMBDA_HARM_<REGION>_A.csv, ..._B.csv
  - output/lambda/LAMBDA_ATTR_<REGION>_A.csv, ..._B.csv
Table 5 (combined):
  - output/tables/Table5_lambda.csv
Figure 4:
  - single region:  output/figures/Fig4_lambda_polar_<REGION>.png
  - multi-region :  output/figures/Fig4_lambda_polar_obs_vs_model.png
 | <module> | heuristic string match
- other | <HERE_PARENT> | <module> | constant definition; attribute placeholder
- table | <OUT_ATTR> | main | variable placeholder
- table | <OUT_CSV> | main | variable placeholder
- table | <OUT_H> | main | variable placeholder
- table | <T5> | main | variable placeholder
- table | <T5> | main | variable placeholder

## scripts/make_cmip_panel.py
Brief: make_cmip_panel.py — 生成 CMIP 月度面板（time, model, region, value），供 feedback_fit 使用。
Constants: CMIP_ROOT = gws/nopw/j04/csgap/ceppi/data/cmip6/5x5, ROOT = <HERE_PARENT>

- other | <HERE_PARENT> | <module> | constant definition; attribute placeholder
- other | gws/nopw/j04/csgap/ceppi/data/cmip6/5x5 | <module> | constant definition; Path constructor
- table | <OUT> | main | variable placeholder

## scripts/make_table1_harmonics.py
Brief: Aggregate harmonics (A, phi) from output/harmonics/ into one tall table:
Constants: ROOT = <HERE_PARENT>

- other | 
Aggregate harmonics (A, phi) from output/harmonics/ into one tall table:
  output/tables/Table1_harmonics_A_phi.csv

It ingests:
  - cllmodis_harmonics_*.csv
  - clswlow_harmonics_*.csv, cllwlow_harmonics_*.csv
  - ERA5_*_harmonics_*.csv
  - CERES_*_harmonics_*.csv   (if present in future)

Expected columns (robust to presence/absence of extras):
  ["source","var","region","span","amplitude","phase_deg","R2","n_valid"]
 | <module> | heuristic string match
- other | <HERE_PARENT> | <module> | constant definition; attribute placeholder
- table | <OUT_CSV> | main | variable placeholder

## scripts/obs_periodogram_test.py
Brief: obs_periodogram_test.py

- figure | <OUT_PNG> | make_figure | variable placeholder
- other | 
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
 | <module> | heuristic string match
- other | <OUT_DIR> | main | variable placeholder; directory creation
- other | <P> | ensure_dirs | variable placeholder; Path constructor; directory creation
- other | output/rednoisefile | main | heuristic string match
- table | <OUT_CSV> | run_one_region | variable placeholder
- table | output/regional_monthly/cllmodis_region_mean_2003-2022_{region}.csv | parse_args | heuristic string match

## scripts/region_mean_ceres.py
Brief: CERES regional monthly means (STRICT, NO GUESSING)
Constants: ROOT = <HERE_PARENT>

- other | <HERE_PARENT> | <module> | constant definition; attribute placeholder
- table | <OUT_CSV> | main | variable placeholder

## scripts/region_mean_era5.py
Brief: ERA5 regional monthly means + harmonics (STRICT, NO GUESSING)
Constants: ROOT = <HERE_PARENT>

- other | <HERE_PARENT> | <module> | constant definition; attribute placeholder
- table | <HARM_PATH> | main | variable placeholder
- table | <OUT_CSV> | main | variable placeholder
- table | <SMALL_PATH> | main | variable placeholder

## scripts/region_mean_modis.py
Brief: Compute regional mean time series, monthly climatology, and first-harmonic fit
Constants: ROOT = <HERE_PARENT>

- log | 
Compute regional mean time series, monthly climatology, and first-harmonic fit
for MODIS/LCRE on Paulo's 5x5 monthly grids.

Outputs (under output/):
  - regional_monthly/<var>_region_mean_<span>_<REGION>.nc
  - regional_monthly/<var>_region_mean_<span>_<REGION>.csv
  - regional_monthly/<var>_climatology_<span>_<REGION>.csv
  - harmonics/<var>_harmonics_<span>_<REGION>.csv

Usage:
  python scripts/region_mean_modis.py --region SEP --var cllmodis
  # var: cllmodis, clswlow, cllwlow
 | <module> | heuristic string match
- other | <HERE_PARENT> | <module> | constant definition; attribute placeholder
- table | <OUT_CSV> | main | variable placeholder
- table | <OUT_CSV_CLIM> | main | variable placeholder
- table | <OUT_CSV_FIT> | main | variable placeholder
- table | <OUT_NC> | main | variable placeholder

## scripts/seasonal_consistency.py
Brief: Seasonal consistency diagnostics (Shape / Amplitude / Phase) + Taylor diagram.
Constants: ROOT = <HERE_PARENT>

- figure | <FIG_PATH> | main | variable placeholder
- log | 
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
 | <module> | heuristic string match
- other | <HERE_PARENT> | <module> | constant definition; attribute placeholder

## scripts/seasonal_consistency_rank.py
Brief: seasonal_consistency_rank.py

- figure | <OUT_PNG> | plot_regionwise | variable placeholder
- figure | <OUT_PNG> | plot_global_top | variable placeholder
- other | 
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
 | <module> | heuristic string match
- other | <FIG_DIR> | main | variable placeholder; directory creation
- other | <P> | ensure_dirs | variable placeholder; Path constructor; directory creation
- other | <TBL_DIR> | main | variable placeholder; directory creation
- other | output/figures | main | heuristic string match
- other | output/figures | parse_args | heuristic string match
- other | output/tables | main | heuristic string match
- other | output/tables | parse_args | heuristic string match
- table | <OUT_CSV> | main | variable placeholder

## scripts/sensitivity_beta.py
Brief: Seasonal (month-by-month) sensitivity:
Constants: ROOT = <HERE_PARENT>

- figure | <FIG_PATH> | main | variable placeholder
- other | 
    Prefer:  output/regional_monthly/MODIS_<REGION>_monthly.csv  (应含列 MODIS_LCF)
    Fallback:cllmodis_region_mean_*_<REGION>.csv   (两列: time, value 或无表头)
    返回: pd.Series(index=datetime, name='LCF')
     | _read_modis_lcf | heuristic string match
- other | 
Seasonal (month-by-month) sensitivity:
  LCF'_m = sum_j beta_j^(m) * CCF'_{j,m} + eps_m

Inputs (per region):
  - output/regional_monthly/MODIS_<REGION>_monthly.csv   (needs column MODIS_LCF; fallback to cllmodis_region_mean_*.csv)
  - output/regional_monthly/ERA5_<REGION>_monthly.csv    (columns like ERA5_EIS, ERA5_W500, ERA5_SST, ERA5_TS, ERA5_U10, ERA5_Q)

Preprocessing:
  - For each calendar month m (1..12), z-score LCF and every CCF using that month's (across years) mean/std
  - Check multicollinearity via VIF (warn if > 5)
  - OLS with Newey–West (HAC) standard errors (lag = --lags)

Outputs:
  - output/sensitivity_beta/BETA_<REGION>.csv
      columns: month,var,beta,se,t,p,ci_low,ci_high,n,rsq
  - output/sensitivity_beta/BETA_HARM_<REGION>.csv
      columns: var,A,phase_deg,peak_month,r2_beta,n_valid
  - output/tables/Table2_beta.csv  (append/update one row per var per region)
      columns: region,var,beta_mean,beta_ci_low,beta_ci_high,A,phase_deg
  - output/figures/Fig2_beta_stripes_<REGION>.png  (variables x 12 months heatmap-like stripes)

Usage:
  python scripts/sensitivity_beta.py --region NEP --ccf EIS,W500,SST,U10,Q --lags 1
  # ΔSST 可写成 DSST 或 ΔSST
 | <module> | heuristic string match
- other | <HERE_PARENT> | <module> | constant definition; attribute placeholder
- table | <OUT_BETA> | main | variable placeholder
- table | <OUT_HARM> | main | variable placeholder
- table | <TABLE2_PATH> | main | variable placeholder
- table | <TABLE2_PATH> | main | variable placeholder

## scripts/utils_climatology.py
Brief: No module docstring detected.


## scripts/utils_config.py
Brief: No module docstring detected.

- log | 
    统一输出目录结构：
      output/
        regional_monthly/  harmonics/  sensitivity_beta/  gamma/  dLCF_dTs/
        lambda/  tables/  figures/  logs/
    允许老配置的 results_dir/figures_dir/logs_dir 共存（向后兼容）。
     | _ensure_output_tree | heuristic string match
- other | <P> | get_outdir | variable placeholder; directory creation
- other | <ROOT>/<V> | _ensure_output_tree | variable placeholder; Path constructor; directory creation
- other | 返回 output/<kind>/<name> 的完整路径。 | make_output_path | heuristic string match

## scripts/utils_region.py
Brief: No module docstring detected.


## Consolidated conventions
- Prefer storing tabular data in `tables/` with descriptive filenames.
- Figures should be written to `fig/` or `figures/` folders with informative names.
- Use `output/` for derived datasets and diagnostics.
- Write execution logs to `logs/` with timestamps or identifiers.
