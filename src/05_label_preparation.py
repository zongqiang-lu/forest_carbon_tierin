"""
AGB标签制备脚本
==================
功能：基于特征栈步骤生成的森林掩膜和AGB连续值，Jenks自然断点分级
仅适用于有AGB标签的年份

输入：04_feature_stack.py生成的 forest_mask.tif + agb_continuous.tif
输出：data/{region}/{year}/aligned/agb_class_{n}level.tif, jenks_breaks.json
"""

import sys
import time
import numpy as np
import rasterio
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import parse_args, RegionConfig


def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def prepare_labels(cfg: RegionConfig):
    print("=" * 60, flush=True)
    print(f"PKU-AGB 标签制备 ({cfg.region_name} {cfg.year})", flush=True)
    print("=" * 60, flush=True)

    if not cfg.has_agb:
        log("该年份无AGB数据，跳过标签制备")
        return None

    cfg.ensure_dirs()

    # 增量执行：已存在则跳过
    if cfg.agb_class.exists() and cfg.jenks_breaks_json.exists() and not cfg.force:
        log(f"AGB分级标签已存在，跳过: {cfg.agb_class}")
        return None

    # 1. 读取04步骤生成的AGB连续值（已裁剪对齐到UTM）
    log(f"[1/3] 读取AGB连续值: {cfg.agb_continuous}")
    t0 = time.time()

    if not cfg.agb_continuous.exists():
        log(f"错误: AGB连续值文件不存在，请先运行04_feature_stack.py")
        return None

    with rasterio.open(cfg.agb_continuous) as src:
        agb_clean = src.read(1).astype(np.float32)
        agb_profile = src.profile.copy()

    log(f"  AGB尺寸: {agb_clean.shape}, 读取耗时: {time.time()-t0:.1f}s")

    # 2. 读取森林掩膜（04步骤已用CLCD+AGB生成）
    log(f"[2/3] 读取森林掩膜: {cfg.forest_mask}")
    with rasterio.open(cfg.forest_mask) as src:
        forest_mask = src.read(1) == 1

    # 应用森林掩膜
    agb_clean = np.where(forest_mask, agb_clean, np.nan)

    valid_vals = agb_clean[np.isfinite(agb_clean)]
    n_forest = int(forest_mask.sum())
    n_valid = len(valid_vals)
    log(f"  森林像元: {n_forest}, 有效AGB: {n_valid}")
    if n_valid > 0:
        log(f"  AGB范围: {valid_vals.min():.2f} ~ {valid_vals.max():.2f}")

    # 3. Jenks分级
    log(f"[3/3] Jenks自然断点分{cfg.n_classes}级...")
    t0 = time.time()

    if n_valid == 0:
        log("错误: 无有效AGB像元，无法分级")
        return None

    MAX_JENKS_SAMPLES = 50000
    if len(valid_vals) > MAX_JENKS_SAMPLES:
        rng = np.random.default_rng(cfg.random_seed)
        sample = rng.choice(valid_vals, size=MAX_JENKS_SAMPLES, replace=False)
    else:
        sample = valid_vals
    log(f"  Jenks采样: {len(sample)} 像元, unique值: {len(np.unique(sample))}")

    import jenkspy
    breaks = jenkspy.jenks_breaks(sample.tolist(), n_classes=cfg.n_classes)
    # AI辅助生成Qwen3.5, 2026-3-22
    log(f"  Jenks断点计算完成, 耗时: {time.time()-t0:.1f}s")

    for i in range(cfg.n_classes):
        log(f"    级别{i+1}: {breaks[i]:.2f} ~ {breaks[i+1]:.2f} t/ha")

    # 新增：过滤分级边界±8t/ha范围内的模糊样本，减少边界混淆
    mask_boundary = np.zeros_like(agb_clean, dtype=bool)
    for i in range(1, len(breaks) - 1):
        boundary_val = breaks[i]
        mask_boundary |= (agb_clean >= boundary_val - 8) & (agb_clean <= boundary_val + 8)
    original_valid = np.sum(np.isfinite(agb_clean))
    agb_clean[mask_boundary] = np.nan
    filtered_count = original_valid - np.sum(np.isfinite(agb_clean))
    log(f"  过滤边界±8t/ha模糊样本: {filtered_count} 个，占比: {filtered_count / original_valid * 100:.1f}%")

    # 分类
    agb_class = np.full_like(agb_clean, np.nan, dtype=np.float32)
    for i in range(cfg.n_classes):
        if i < cfg.n_classes - 1:
            mask_c = (agb_clean >= breaks[i]) & (agb_clean < breaks[i+1])
        else:
            mask_c = (agb_clean >= breaks[i]) & (agb_clean <= breaks[i+1])
        agb_class[mask_c] = i + 1

    # 保存
    class_profile = agb_profile.copy()
    class_profile.update({"dtype": "float32", "nodata": np.nan, "compress": "lzw"})

    with rasterio.open(cfg.agb_class, "w", **class_profile) as dst:
        dst.write(agb_class, 1)
    log(f"  分级输出: {cfg.agb_class.name}")

    # 4. 保存断点信息
    log("保存断点信息...")
    class_labels_cn = {
        3: ["低碳密度", "中碳密度", "高碳密度"],
        5: ["低碳密度", "较低碳密度", "中碳密度", "较高碳密度", "高碳密度"],
    }.get(cfg.n_classes, [f"级别{i+1}" for i in range(cfg.n_classes)])
    breaks_info = {
        "n_classes": cfg.n_classes,
        "method": "Jenks Natural Breaks (jenkspy)",
        "breaks": [float(b) for b in breaks],
        "class_labels": class_labels_cn[:cfg.n_classes],
        "class_ranges": [f"{breaks[i]:.2f} ~ {breaks[i+1]:.2f}" for i in range(cfg.n_classes)],
        "agb_nodata": cfg.agb_nodata,
        "agb_max_valid": cfg.agb_max_valid,
        "forest_mask_source": "CLCD (value=2) AND AGB range filter",
        "boundary_filter": "±8 t/ha around Jenks breakpoints removed",
    }
    with open(cfg.jenks_breaks_json, "w", encoding="utf-8") as f:
        json.dump(breaks_info, f, ensure_ascii=False, indent=2)
    log(f"  断点信息: {cfg.jenks_breaks_json.name}")

    print("\n" + "=" * 60, flush=True)
    print("标签制备完成!", flush=True)
    print("=" * 60, flush=True)
    return breaks_info


if __name__ == "__main__":
    cfg = parse_args()
    prepare_labels(cfg)
