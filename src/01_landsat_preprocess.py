"""
Landsat L2SP 多景预处理脚本
===========================
功能：自动发现多景Landsat数据、逐景裁剪缩放掩膜、中值合成无云影像、输出7波段堆叠TIF

合成策略：逐波段pixel-based median composite
- 对云和异常值鲁棒 (Hansen et al., 2013)
- 保留光谱信息，不丢失细节
- nanmedian自动忽略NaN像元

参考：
- USGS Landsat Collection 2 Level-2 Data Product Guide
- L2SP SR缩放: SR = DN × 0.0000275 - 0.2
- 不掩膜气溶胶：LaSRC已做大气校正
"""

import sys
import time
import glob
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
import geopandas as gpd
from pathlib import Path

# 添加src目录到路径以导入config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import parse_args, RegionConfig

# SR缩放因子 (USGS Collection 2 L2SP)
SR_SCALE = 0.0000275
SR_OFFSET = -0.2

# Landsat 8/9 OLI 光谱波段
SR_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
SR_DESCRIPTIONS = [
    "Coastal/Aerosol", "Blue", "Green", "Red",
    "NIR", "SWIR1", "SWIR2"
]


def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def find_band_path(scene_dir: Path, band_name: str) -> str:
    """查找指定波段的文件路径"""
    pattern = str(scene_dir / f"*_{band_name}.TIF")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"未找到波段文件: {band_name}, 搜索路径: {pattern}")
    return matches[0]


def find_scenes(landsat_dir: Path) -> list:
    """自动发现所有Landsat场景目录"""
    scenes = sorted([
        d for d in landsat_dir.iterdir()
        if d.is_dir() and d.name.startswith("LC0")
    ])
    return scenes


def parse_qa_pixel(qa_arr):
    """
    解析QA_PIXEL波段，生成云/云影/雪掩膜
    bit 0: Fill, bit 1: Dilated Cloud, bit 3: Cloud,
    bit 4: Cloud Shadow, bit 5: Snow/Ice
    返回: True=该像元应被掩膜(无效)
    """
    fill = (qa_arr & (1 << 0)) > 0
    dilated_cloud = (qa_arr & (1 << 1)) > 0
    cloud = (qa_arr & (1 << 3)) > 0
    cloud_shadow = (qa_arr & (1 << 4)) > 0
    snow = (qa_arr & (1 << 5)) > 0
    return fill | cloud | cloud_shadow | snow | dilated_cloud


def parse_qa_radsat(qa_arr):
    """解析QA_RADSAT波段，检测饱和像元"""
    saturated = np.zeros(qa_arr.shape, dtype=bool)
    for bit in range(1, 8):
        saturated |= (qa_arr & (1 << bit)) > 0
    return saturated


def clip_with_boundary(dataset, gdf):
    """使用边界裁剪影像"""
    try:
        out_image, out_transform = rio_mask(
            dataset, gdf.geometry.tolist(), crop=True, nodata=0
        )
        return out_image, out_transform
    except Exception as e:
        print(f"  裁剪失败: {e}, 将使用全影像", flush=True)
        return None, None


def process_single_scene(scene_dir: Path, gdf, sr_bands=SR_BANDS):
    """
    处理单个Landsat场景：裁剪+缩放+掩膜

    返回:
        sr_stack: (n_bands, height, width) float32, 无效为NaN
        qa_mask: (height, width) bool, True=有效
        profile: rasterio profile
        stats: dict 统计信息
    """
    scene_name = scene_dir.name
    log(f"  处理场景: {scene_name}")

    # 裁剪第一波段获取参考网格
    ref_path = find_band_path(scene_dir, sr_bands[0])
    with rasterio.open(ref_path) as src:
        clipped, clipped_transform = clip_with_boundary(src, gdf)
        if clipped is None:
            data = src.read(1).astype(np.float32)
            out_transform = src.transform
            out_profile = src.profile.copy()
        else:
            data = clipped[0].astype(np.float32)
            out_transform = clipped_transform
            out_profile = src.profile.copy()
            out_profile.update({
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": out_transform,
            })

    shape = data.shape
    n_bands = len(sr_bands)
    sr_stack = np.full((n_bands, shape[0], shape[1]), np.nan, dtype=np.float32)

    # 处理第一个波段
    valid_dn = data > 0
    sr = np.where(valid_dn, data * SR_SCALE + SR_OFFSET, np.nan)
    sr[(sr <= 0) | (sr > 1)] = np.nan
    sr_stack[0] = sr

    # 处理剩余波段
    for i, band_name in enumerate(sr_bands[1:], 1):
        band_path = find_band_path(scene_dir, band_name)
        with rasterio.open(band_path) as src:
            clipped, _ = clip_with_boundary(src, gdf)
            if clipped is None:
                data = src.read(1).astype(np.float32)
            else:
                data = clipped[0].astype(np.float32)

        valid_dn = data > 0
        sr = np.where(valid_dn, data * SR_SCALE + SR_OFFSET, np.nan)
        sr[(sr <= 0) | (sr > 1)] = np.nan
        sr_stack[i] = sr

    # QA掩膜
    qa_pixel_path = find_band_path(scene_dir, "QA_PIXEL")
    qa_radsat_path = find_band_path(scene_dir, "QA_RADSAT")

    with rasterio.open(qa_pixel_path) as src:
        clipped_qa, _ = clip_with_boundary(src, gdf)
        qa_pixel = clipped_qa[0] if clipped_qa is not None else src.read(1)

    with rasterio.open(qa_radsat_path) as src:
        clipped_qa, _ = clip_with_boundary(src, gdf)
        qa_radsat = clipped_qa[0] if clipped_qa is not None else src.read(1)

    cloud_mask = parse_qa_pixel(qa_pixel)
    sat_mask = parse_qa_radsat(qa_radsat)
    bad_mask = cloud_mask | sat_mask  # True=无效

    # 统计
    total_px = shape[0] * shape[1]
    cloud_pct = 100 * cloud_mask.sum() / total_px
    sat_pct = 100 * sat_mask.sum() / total_px

    # 应用掩膜：无效像元设为NaN
    for i in range(n_bands):
        sr_stack[i][bad_mask] = np.nan

    valid_after = np.isfinite(sr_stack[0]).sum()
    valid_pct = 100 * valid_after / total_px

    stats = {
        "scene": scene_name,
        "cloud_pct": cloud_pct,
        "sat_pct": sat_pct,
        "valid_pct": valid_pct,
    }
    log(f"    云: {cloud_pct:.1f}%, 饱和: {sat_pct:.1f}%, 有效: {valid_pct:.1f}%")

    # 更新profile
    out_profile.update({
        "count": n_bands,
        "dtype": "float32",
        "nodata": np.nan,
        "compress": "lzw",
    })

    return sr_stack, out_profile, stats


def median_composite(scene_stacks: list) -> np.ndarray:
    """
    多景中值合成

    Args:
        scene_stacks: list of (n_bands, height, width) arrays

    Returns:
        composite: (n_bands, height, width) median composite
    """
    stacked = np.stack(scene_stacks, axis=0)  # (n_scenes, n_bands, h, w)
    # 逐波段取nanmedian
    n_scenes, n_bands, h, w = stacked.shape
    composite = np.full((n_bands, h, w), np.nan, dtype=np.float32)
    for b in range(n_bands):
        composite[b] = np.nanmedian(stacked[:, b, :, :], axis=0)
    return composite


def preprocess_landsat(cfg: RegionConfig):
    """主处理流程：多景Landsat预处理+中值合成"""
    print("=" * 60, flush=True)
    print("Landsat L2SP 多景预处理与中值合成", flush=True)
    print("=" * 60, flush=True)

    cfg.ensure_dirs()

    # 增量执行：已存在则跳过
    if cfg.landsat_sr_stack.exists() and not cfg.force:
        log(f"Landsat SR栈已存在，跳过: {cfg.landsat_sr_stack}")
        return

    # 加载边界
    gdf = gpd.read_file(cfg.boundary)
    log(f"研究区: {cfg.region_name} ({cfg.region_id})")
    log(f"边界: {cfg.boundary.name}, CRS: {gdf.crs}")

    # 发现场景
    scenes = find_scenes(cfg.landsat_dir)
    if not scenes:
        log(f"错误: 未找到Landsat场景 ({cfg.landsat_dir})")
        return
    log(f"发现 {len(scenes)} 景数据:")
    for s in scenes:
        log(f"  {s.name}")

    # 逐景处理
    all_stacks = []
    all_stats = []
    ref_profile = None

    for i, scene_dir in enumerate(scenes):
        log(f"\n[{i+1}/{len(scenes)}] 处理场景: {scene_dir.name}")
        sr_stack, profile, stats = process_single_scene(scene_dir, gdf)
        all_stacks.append(sr_stack)
        all_stats.append(stats)
        if ref_profile is None:
            ref_profile = profile

    # 中值合成
    if len(all_stacks) == 1:
        log("\n仅1景数据，无需合成")
        composite = all_stacks[0]
    else:
        log(f"\n中值合成 {len(all_stacks)} 景数据...")
        t0 = time.time()
        composite = median_composite(all_stacks)
        log(f"  合成耗时: {time.time()-t0:.1f}s")

    # 统计合成结果
    total_px = composite.shape[1] * composite.shape[2]
    for b, (band_name, desc) in enumerate(zip(SR_BANDS, SR_DESCRIPTIONS)):
        valid = np.isfinite(composite[b]).sum()
        log(f"  {band_name} ({desc}): 有效 {valid} ({100*valid/total_px:.1f}%)")

    # 输出7波段堆叠TIF
    log(f"\n输出: {cfg.landsat_sr_stack}")
    out_profile = ref_profile.copy()
    out_profile.update({
        "count": len(SR_BANDS),
        "dtype": "float32",
        "nodata": np.nan,
        "compress": "lzw",
    })
    with rasterio.open(cfg.landsat_sr_stack, "w", **out_profile) as dst:
        dst.write(composite)
        # 写入波段描述
        for b, desc in enumerate(SR_DESCRIPTIONS, 1):
            dst.set_band_description(b, desc)
        dst.descriptions = tuple(SR_DESCRIPTIONS)

    # 生成综合质量掩膜 (1=有效, 0=无效)
    log("生成质量掩膜...")
    all_valid = np.isfinite(composite).all(axis=0)  # 所有波段都有效的像元
    valid_pct = 100 * all_valid.sum() / total_px
    log(f"  综合有效像元: {all_valid.sum()} / {total_px} ({valid_pct:.1f}%)")

    mask_profile = out_profile.copy()
    mask_profile.update({
        "count": 1,
        "dtype": "uint8",
        "nodata": 255,
        "compress": "lzw",
    })
    with rasterio.open(cfg.quality_mask, "w", **mask_profile) as dst:
        dst.write(all_valid.astype(np.uint8), 1)

    # 打印汇总
    print("\n" + "=" * 60, flush=True)
    print("Landsat L2SP 多景预处理完成!", flush=True)
    print(f"输出: {cfg.landsat_sr_stack} ({len(SR_BANDS)}波段)")
    print(f"掩膜: {cfg.quality_mask}")
    for s in all_stats:
        print(f"  场景 {s['scene']}: 云{s['cloud_pct']:.1f}%, 有效{s['valid_pct']:.1f}%")
    print(f"合成后有效: {valid_pct:.1f}%")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    cfg = parse_args()
    preprocess_landsat(cfg)
