"""
特征栈构建脚本
==============
功能：从landsat_sr_stack(7波段)、indices_stack(13波段)、terrain_stack(8波段)
      读取所有特征，对齐裁剪至研究区范围，应用森林掩膜(CLCD)+质量掩膜，
      输出feature_stack.tif(28波段)

特征来源：
  landsat_sr_stack.tif: B1, B2, B3, B4, B5, B6, B7 (7波段)
  indices_stack.tif:    NDVI, EVI, NDMI, SAVI, NDWI, MSAVI, NBR, SR_B5B4, SR_B6B5, SR_B7B5, TCB, TCG, TCW (13波段)
  terrain_stack.tif:    elevation, slope, aspect, twi, pcurv, ccurv, slope_pos, roughness (8波段)

森林掩膜：
  使用CLCD数据集（值=2为森林），Albers投影需重投影到UTM
  有AGB年份：CLCD=2 AND AGB在合理范围(0 < AGB <= 500)
  无AGB年份：仅CLCD=2
"""

import sys
import time
import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
from rasterio.mask import mask as rio_mask
import geopandas as gpd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import parse_args, RegionConfig, PROJECT_ROOT

# 特征定义：每个stack文件中的波段名
LANDSAT_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
INDEX_BANDS = [
    "NDVI", "EVI", "NDMI", "SAVI", "NDWI",
    "MSAVI", "NBR", "SR_B5B4", "SR_B6B5", "SR_B7B5",
    "TCB", "TCG", "TCW",
]
TERRAIN_BANDS = [
    "elevation", "slope", "aspect", "twi",
    "pcurv", "ccurv", "slope_pos", "roughness",
]


def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def align_raster_to_target(src_path, target_transform, target_crs, target_shape,
                           band_indices=None, target_nodata=np.nan):
    """
    将源栅格重投影/重采样到目标网格（使用WarpedVRT优化）
    """
    from rasterio.vrt import WarpedVRT

    with rasterio.open(src_path) as src:
        if band_indices is None:
            band_indices = list(range(1, src.count + 1))

        src_crs = src.crs
        src_transform = src.transform
        needs_reproject = (src_crs != target_crs) or (src_transform != target_transform)

        if not needs_reproject:
            aligned_bands = []
            for bi in band_indices:
                band_data = src.read(bi)
                aligned_bands.append(band_data.astype(np.float32))
            return aligned_bands

        # 使用WarpedVRT进行重投影（比逐波段reproject更快）
        vrt_params = {
            "crs": target_crs,
            "transform": target_transform,
            "height": target_shape[0],
            "width": target_shape[1],
            "nodata": target_nodata,
            "resampling": Resampling.bilinear,
        }

        with WarpedVRT(src, **vrt_params) as vrt:
            data = vrt.read(band_indices)
            aligned_bands = [data[i].copy() for i in range(len(band_indices))]

    return aligned_bands


def create_target_grid_from_boundary(boundary_path, target_crs, target_res):
    """
    从边界文件创建目标网格（用于无AGB年份）
    """
    gdf = gpd.read_file(boundary_path)
    if gdf.crs is None:
        raise ValueError(f"边界文件缺少坐标系: {boundary_path}")
    gdf_utm = gdf.to_crs(target_crs)
    bounds = gdf_utm.total_bounds  # [minx, miny, maxx, maxy]

    minx = np.floor(bounds[0] / target_res) * target_res
    miny = np.floor(bounds[1] / target_res) * target_res
    maxx = np.ceil(bounds[2] / target_res) * target_res
    maxy = np.ceil(bounds[3] / target_res) * target_res

    width = int((maxx - minx) / target_res)
    height = int((maxy - miny) / target_res)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    return transform, (height, width), gdf_utm


def prepare_clcd_forest_mask(clcd_path, boundary_path, target_crs, target_res,
                              target_transform=None, target_shape=None,
                              cache_path=None):
    """
    从CLCD数据提取森林掩膜
    CLCD为Albers投影，需重投影到UTM并裁剪到研究区
    值=2表示森林
    
    支持缓存：如果cache_path存在且源CLCD未变，直接读取缓存
    """
    # 检查缓存
    if cache_path is not None and cache_path.exists():
        log(f"  读取CLCD缓存: {cache_path}")
        with rasterio.open(cache_path) as src:
            cached_data = src.read(1)
            forest_mask = cached_data == 1  # 缓存中1=森林
        n_forest = int(forest_mask.sum())
        n_total = forest_mask.size
        log(f"  缓存命中: CLCD森林像元 {n_forest} / {n_total} ({100*n_forest/n_total:.1f}%)")
        return forest_mask

    log(f"  读取CLCD: {clcd_path}")

    # 读取边界
    gdf = gpd.read_file(boundary_path)
    gdf_utm = gdf.to_crs(target_crs)
    bounds = gdf_utm.total_bounds

    # 先裁剪CLCD到研究区附近（避免处理全国数据）
    with rasterio.open(clcd_path) as src:
        log(f"  CLCD原始CRS: {src.crs}, 尺寸: {src.shape}")

        # 将UTM边界转换到CLCD的Albers坐标系获取裁剪范围
        gdf_albers = gdf.to_crs(src.crs)
        albers_bounds = gdf_albers.total_bounds

        # 加buffer确保覆盖
        buffer = 5000  # 5km buffer
        window = src.window(
            albers_bounds[0] - buffer, albers_bounds[1] - buffer,
            albers_bounds[2] + buffer, albers_bounds[3] + buffer
        )

        # 读取裁剪后的CLCD
        clcd_data = src.read(1, window=window, boundless=True, fill_value=0)
        clcd_transform = src.window_transform(window)
        clcd_crs = src.crs

    log(f"  CLCD裁剪后尺寸: {clcd_data.shape}")

    # 重投影到UTM
    if target_transform is None or target_shape is None:
        target_transform, target_shape, _ = create_target_grid_from_boundary(
            boundary_path, target_crs, target_res
        )

    clcd_aligned = np.full(target_shape, 0, dtype=np.float32)
    reproject(
        source=clcd_data.astype(np.float32),
        destination=clcd_aligned,
        src_transform=clcd_transform,
        src_crs=clcd_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest,  # 分类数据用最近邻
        src_nodata=0,
        dst_nodata=0,
    )

    # 森林掩膜：CLCD值=2
    forest_mask = clcd_aligned == 2
    n_forest = int(forest_mask.sum())
    n_total = forest_mask.size
    log(f"  CLCD森林像元: {n_forest} / {n_total} ({100*n_forest/n_total:.1f}%)")

    # 保存缓存
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        mask_profile = {
            "driver": "GTiff", "dtype": "uint8", "nodata": 255,
            "width": target_shape[1], "height": target_shape[0],
            "count": 1, "crs": target_crs, "transform": target_transform,
            "compress": "lzw",
        }
        with rasterio.open(cache_path, "w", **mask_profile) as dst:
            dst.write(forest_mask.astype(np.uint8), 1)
        log(f"  CLCD缓存已保存: {cache_path}")

    del clcd_data, clcd_aligned
    return forest_mask


def build_feature_stack(cfg: RegionConfig):
    """主处理流程：构建特征栈"""
    print("=" * 60, flush=True)
    print(f"特征栈构建 ({cfg.region_name} {cfg.year})", flush=True)
    print("=" * 60, flush=True)

    cfg.ensure_dirs()

    # 增量执行：已存在则跳过（除非--force）
    if cfg.feature_stack.exists() and cfg.valid_mask.exists() and not cfg.force:
        log(f"特征栈已存在，跳过: {cfg.feature_stack}")
        return

    # 1. 确定目标网格
    agb_data = None
    if cfg.has_agb:
        log(f"\n[1/6] 确定目标网格 (以AGB为参考): {cfg.agb}")
        with rasterio.open(cfg.agb) as src:
            target_crs = src.crs
            target_transform = src.transform
            target_shape = (src.height, src.width)
            agb_data = src.read(1).astype(np.float32)
            target_profile = src.profile.copy()
        log(f"  CRS: {target_crs}, 尺寸: {target_shape}, 分辨率: {src.res}")
    else:
        # 无AGB年份（预测模式）：使用训练年份(2019)的feature_stack作为参考网格
        # 确保预测年特征与训练年空间对齐，模型才能正确推理
        ref_feature_stack = (PROJECT_ROOT / "data" / cfg.region_id
                             / str(cfg.model_year) / "aligned" / "feature_stack.tif")
        if ref_feature_stack.exists():
            log(f"\n[1/6] 确定目标网格 (以{cfg.model_year}年feature_stack为参考, 预测模式)")
            with rasterio.open(ref_feature_stack) as ref_src:
                target_crs = ref_src.crs
                target_transform = ref_src.transform
                target_shape = (ref_src.height, ref_src.width)
                target_profile = ref_src.profile.copy()
            log(f"  CRS: {target_crs}, 尺寸: {target_shape}")
        else:
            log(f"\n[1/6] 确定目标网格 (以边界为参考, 无AGB且无{cfg.model_year}年参考)")
            target_transform, target_shape, _ = create_target_grid_from_boundary(
                cfg.boundary, cfg.crs, cfg.resolution
            )
            target_crs = cfg.crs
            target_profile = {
                "driver": "GTiff", "dtype": "float32", "nodata": np.nan,
                "width": target_shape[1], "height": target_shape[0],
                "count": 1, "crs": target_crs, "transform": target_transform,
                "compress": "lzw",
            }
            log(f"  CRS: {target_crs}, 尺寸: {target_shape}")

    # 2. 读取质量掩膜
    log(f"\n[2/6] 读取质量掩膜: {cfg.quality_mask}")
    quality_bands = align_raster_to_target(
        cfg.quality_mask, target_transform, target_crs, target_shape,
        band_indices=[1], target_nodata=0
    )
    quality_valid = quality_bands[0] == 1
    log(f"  Landsat质量掩膜有效像元: {quality_valid.sum()} / {quality_valid.size}")
    del quality_bands

    # 3. 生成森林掩膜 (CLCD)
    log(f"\n[3/6] 生成森林掩膜 (CLCD值=2为森林)...")
    forest_mask = prepare_clcd_forest_mask(
        cfg.clcd, cfg.boundary, target_crs, cfg.resolution,
        target_transform, target_shape,
        cache_path=cfg.clcd_forest_mask_cache
    )

    # 有AGB年份叠加AGB过滤
    if cfg.has_agb and agb_data is not None:
        log(f"  {cfg.year}年叠加AGB过滤 (0 < AGB <= {cfg.agb_max_valid})...")
        if target_profile.get("driver") == "GTiff" and agb_data.shape == target_shape:
            agb_aligned = agb_data
        else:
            agb_bands = align_raster_to_target(
                cfg.agb, target_transform, target_crs, target_shape,
                band_indices=[1], target_nodata=np.nan
            )
            agb_aligned = agb_bands[0]
            del agb_bands

        agb_valid = (agb_aligned > cfg.agb_nodata) & (agb_aligned <= cfg.agb_max_valid)
        n_before = int(forest_mask.sum())
        forest_mask = forest_mask & agb_valid
        n_after = int(forest_mask.sum())
        log(f"  AGB过滤: {n_before} → {n_after} (剔除{n_before-n_after})")
        del agb_aligned

    log(f"  最终森林像元: {forest_mask.sum()} / {forest_mask.size}")

    # 4. 一次性读取terrain和indices用于质量过滤（复用数据避免重复重投影）
    log("\n[4/6] 读取地形和植被指数（用于质量过滤）...")

    log(f"  读取地形因子: {cfg.terrain_stack}")
    terrain_bands = align_raster_to_target(
        cfg.terrain_stack, target_transform, target_crs, target_shape
    )
    slope_data = terrain_bands[1]  # slope是第2波段

    log(f"  读取植被指数: {cfg.indices_stack}")
    indices_bands = align_raster_to_target(
        cfg.indices_stack, target_transform, target_crs, target_shape
    )
    ndvi_data = indices_bands[0]  # NDVI是第1波段

    # 坡度过滤
    slope_valid = np.isfinite(slope_data) & (slope_data <= cfg.slope_max)
    n_slope_filtered = int((forest_mask & quality_valid & ~slope_valid).sum())
    log(f"  坡度阈值: <={cfg.slope_max}°, 剔除陡坡像元: {n_slope_filtered}")

    # NDVI过滤
    ndvi_valid = np.isfinite(ndvi_data) & (ndvi_data >= cfg.ndvi_min)
    n_ndvi_filtered = int((forest_mask & quality_valid & slope_valid & ~ndvi_valid).sum())
    log(f"  NDVI阈值: >={cfg.ndvi_min}, 剔除低覆盖像元: {n_ndvi_filtered}")

    # 综合有效掩膜 (4重过滤)
    valid_mask = forest_mask & quality_valid & slope_valid & ndvi_valid
    valid_pct = 100 * valid_mask.sum() / valid_mask.size
    n_before = int((forest_mask & quality_valid).sum())
    n_after = int(valid_mask.sum())
    log(f"  质量筛选前: {n_before} → 筛选后: {n_after} (剔除{n_before-n_after}, 保留{100*n_after/n_before:.1f}%)")
    log(f"  综合有效像元: {valid_mask.sum()} / {valid_mask.size} ({valid_pct:.1f}%)")

    # 保存掩膜
    mask_profile = target_profile.copy()
    mask_profile.update({"dtype": "uint8", "nodata": 255, "compress": "lzw"})

    with rasterio.open(cfg.valid_mask, "w", **mask_profile) as dst:
        dst.write(valid_mask.astype(np.uint8), 1)

    with rasterio.open(cfg.forest_mask, "w", **mask_profile) as dst:
        dst.write(forest_mask.astype(np.uint8), 1)

    # 保存质量筛选掩膜(仅quality+坡度+NDVI过滤，不含森林过滤)
    quality_filtered = quality_valid & slope_valid & ndvi_valid
    with rasterio.open(cfg.quality_mask_filtered, "w", **mask_profile) as dst:
        dst.write(quality_filtered.astype(np.uint8), 1)

    del quality_filtered, slope_data, ndvi_data

    # 5. 构建特征栈（复用已读取的terrain_bands和indices_bands）
    log("\n[5/6] 构建特征栈（复用已对齐数据）...")

    feature_names = []
    feature_stack = []

    # --- Landsat SR (7波段) ---
    log(f"  读取Landsat SR: {cfg.landsat_sr_stack}")
    ls_bands = align_raster_to_target(
        cfg.landsat_sr_stack, target_transform, target_crs, target_shape
    )
    for i, name in enumerate(LANDSAT_BANDS):
        band = ls_bands[i]
        band[~valid_mask] = np.nan
        feature_names.append(name)
        feature_stack.append(band)
        valid_px = band[np.isfinite(band)]
        if len(valid_px) > 0:
            log(f"    {name}: range=[{valid_px.min():.4f}, {valid_px.max():.4f}]")
    del ls_bands

    # --- 植被指数 (13波段) - 复用已读取的indices_bands ---
    log(f"  复用植被指数: {cfg.indices_stack}")
    for i, name in enumerate(INDEX_BANDS):
        band = indices_bands[i]
        band[~valid_mask] = np.nan
        feature_names.append(name)
        feature_stack.append(band)
        valid_px = band[np.isfinite(band)]
        if len(valid_px) > 0:
            log(f"    {name}: range=[{valid_px.min():.4f}, {valid_px.max():.4f}]")
    del indices_bands

    # --- 地形因子 (8波段) - 复用已读取的terrain_bands ---
    log(f"  复用地形因子: {cfg.terrain_stack}")
    for i, name in enumerate(TERRAIN_BANDS):
        band = terrain_bands[i]
        band[~valid_mask] = np.nan
        feature_names.append(name)
        feature_stack.append(band)
        valid_px = band[np.isfinite(band)]
        if len(valid_px) > 0:
            log(f"    {name}: range=[{valid_px.min():.4f}, {valid_px.max():.4f}]")
    del terrain_bands

    # 堆叠
    stack = np.stack(feature_stack, axis=0)
    log(f"\n  特征栈形状: {stack.shape} (n_features, height, width)")
    log(f"  特征列表: {feature_names}")

    # 新增：低质量样本过滤 - 剔除所有特征与AGB相关性<0.1的样本
    if cfg.has_agb and agb_data is not None:
        log("\n[样本质量过滤] 计算特征与AGB相关性，剔除相关性<0.1的样本...")
        valid_idx = np.where(valid_mask)
        if len(valid_idx[0]) > 0:
            features_valid = stack[:, valid_idx[0], valid_idx[1]].T  # (n_samples, n_features)
            agb_valid = agb_data[valid_idx[0], valid_idx[1]]
            # 计算每个特征与AGB的皮尔逊相关性
            correlations = np.zeros(features_valid.shape[1])
            for i in range(features_valid.shape[1]):
                mask = np.isfinite(features_valid[:, i]) & np.isfinite(agb_valid)
                if np.sum(mask) > 100:
                    correlations[i] = np.corrcoef(
                        features_valid[mask, i], agb_valid[mask]
                    )[0, 1]
                else:
                    correlations[i] = 0
            # 逐像元检查：该像元的每个有效特征是否与AGB低相关
            # 判断哪些特征是低相关的（全局|correlation| < 0.1）
            low_corr_features = np.abs(correlations) < 0.1
            # 对每个像元：如果所有有效特征都是低相关特征，则判定为低质量
            per_pixel_low = np.ones(len(valid_idx[0]), dtype=bool)
            for i in range(features_valid.shape[1]):
                if not low_corr_features[i]:
                    # 存在至少一个高相关特征，标记这些像元中该特征有效的为非低质量
                    feat_valid = np.isfinite(features_valid[:, i])
                    per_pixel_low[feat_valid] = False
            filtered_samples = np.sum(per_pixel_low)
            valid_mask[valid_idx[0][per_pixel_low], valid_idx[1][per_pixel_low]] = False
            log(f"  过滤低质量样本: {filtered_samples} 个，占比: {filtered_samples / len(valid_idx[0]) * 100:.1f}%")
            # 更新所有特征的无效值
            for i in range(stack.shape[0]):
                stack[i][~valid_mask] = np.nan

    # 保存特征栈
    stack_profile = target_profile.copy()
    stack_profile.update({
        "count": len(feature_names),
        "dtype": "float32",
        "nodata": np.nan,
        "compress": "lzw",
    })
    with rasterio.open(cfg.feature_stack, "w", **stack_profile) as dst:
        dst.write(stack)
        for b, name in enumerate(feature_names, 1):
            dst.set_band_description(b, name)

    # 保存特征名称
    meta = {"feature_names": feature_names, "n_features": len(feature_names)}
    with open(cfg.feature_names_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 保存AGB连续值（仅有AGB年份时）
    log("\n[6/6] 保存AGB连续值...")
    if cfg.has_agb and agb_data is not None:
        if agb_data.shape == target_shape:
            agb_clean = np.where(valid_mask, agb_data, np.nan)
        else:
            agb_bands = align_raster_to_target(
                cfg.agb, target_transform, target_crs, target_shape,
                band_indices=[1], target_nodata=np.nan
            )
            agb_clean = np.where(valid_mask, agb_bands[0], np.nan)
            del agb_bands
        agb_profile = target_profile.copy()
        agb_profile.update({"dtype": "float32", "nodata": np.nan, "compress": "lzw"})
        with rasterio.open(cfg.agb_continuous, "w", **agb_profile) as dst:
            dst.write(agb_clean, 1)

    print("\n" + "=" * 60, flush=True)
    print("特征栈构建完成!")
    print(f"输出: {cfg.feature_stack} ({len(feature_names)}波段)")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    cfg = parse_args()
    build_feature_stack(cfg)
