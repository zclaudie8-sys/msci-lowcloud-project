{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "62d08990-6ed4-4cec-8f07-7334d6ce7666",
   "metadata": {},
   "outputs": [],
   "source": [
    "from utils_config import load_config, get_paths, get_regions, make_output_path, setup_logger, summarize_config\n",
    "\n",
    "# 1) 读配置 & 日志\n",
    "cfg = load_config(\"configs/config.yaml\")\n",
    "logger = setup_logger(cfg, name=\"batch_modis_lcf\")\n",
    "\n",
    "logger.info(\"\\n\" + summarize_config(cfg))\n",
    "\n",
    "# 2) 拿到路径和区域\n",
    "paths = get_paths(cfg)\n",
    "regions = get_regions(cfg)                 # 全部区域\n",
    "sep_box = get_regions(cfg, [\"SEP\"])[\"SEP\"] # 指定一个区域：[-90, -60, -140, -70]\n",
    "\n",
    "logger.info(f\"MODIS path: {paths['modis']}\")\n",
    "logger.info(f\"SEP box: {sep_box} (lat_min, lat_max, lon_min, lon_max)\")\n",
    "\n",
    "# 3) 生成标准输出文件名\n",
    "out_nc  = make_output_path(cfg, stem=\"modis_lcf_monthly\", region=\"SEP\", subdir=\"results\", ext=\"nc\")\n",
    "fig_png = make_output_path(cfg, stem=\"lcf_climatology\",    region=\"SEP\", subdir=\"figures\", ext=\"png\")\n",
    "\n",
    "logger.info(f\"Will save dataset to: {out_nc}\")\n",
    "logger.info(f\"Will save figure  to: {fig_png}\")\n",
    "\n",
    "# 4) ……你的处理代码（读数据 → 计算 → 保存）……\n",
    "# ds.to_netcdf(out_nc)\n",
    "# fig.savefig(fig_png, dpi=200, bbox_inches=\"tight\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 + Jaspy",
   "language": "python",
   "name": "jaspy"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
