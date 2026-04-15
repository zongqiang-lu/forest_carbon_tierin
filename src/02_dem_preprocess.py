"""
DEM 预处理脚本
==============
功能：DEM重投影、重采样至30m、提取海拔/坡度/坡向/TWI/曲率/坡位/粗糙度，输出8波段terrain_stack.tif

参考：
- TWI = ln(As / tan(β))，As为汇水面积，β为坡度
- 曲率：平面曲率和剖面曲率，反映地形凹凸性
- 坡位：相对高程位置（山脊/中坡/谷底）
- DEM重投影采用双线性内插
"""

import sys
import time
import shutil
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import geopandas as gpd
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import (
    gaussian_filter,
    uniform_filter,
    maximum_filter,
    minimum_filter,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import parse_args, RegionConfig

TERRAIN_BANDS = [
    "elevation",
    "slope",
    "aspect",
    "twi",
    "pcurv",
    "ccurv",
    "slope_pos",
    "roughness",
]
TERRAIN_DESCRIPTIONS = [
    "Elevation (m)",
    "Slope (deg)",
    "Aspect (deg)",
    "TWI",
    "Profile Curvature",
    "Plan Curvature",
    "Slope Position",
    "Roughness",
]


def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def reproject_dem(dem_path, target_crs, target_res, boundary_gdf):
    """重投影DEM到目标坐标系并重采样"""
    log(f"  读取DEM: {dem_path.name}")

    with rasterio.open(dem_path) as src:
        log(f"  原始CRS: {src.crs}, 分辨率: {src.res}")

        if boundary_gdf.crs != target_crs:
            boundary_gdf = boundary_gdf.to_crs(target_crs)

        bounds = boundary_gdf.total_bounds
        minx = np.floor(bounds[0] / target_res) * target_res
        miny = np.floor(bounds[1] / target_res) * target_res
        maxx = np.ceil(bounds[2] / target_res) * target_res
        maxy = np.ceil(bounds[3] / target_res) * target_res

        width = int((maxx - minx) / target_res)
        height = int((maxy - miny) / target_res)
        dst_transform = from_bounds(minx, miny, maxx, maxy, width, height)

        dst_profile = src.profile.copy()
        dst_profile.update({
            "crs": target_crs,
            "transform": dst_transform,
            "width": width,
            "height": height,
            "dtype": "float32",
            "nodata": np.nan,
            "compress": "lzw",
        })

        dem_reprojected = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_reprojected,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
        log(f"  重投影完成: {width}x{height}, 分辨率 {target_res}m")

    return dem_reprojected, dst_profile


def clip_with_boundary(data, profile, boundary_gdf):
    """用边界裁剪"""
    from rasterio.io import MemoryFile
    from rasterio.mask import mask as rio_mask

    with MemoryFile() as memfile:
        with memfile.open(**profile) as mem:
            mem.write(data, 1)
        with memfile.open() as mem:
            out_image, out_transform = rio_mask(
                mem, boundary_gdf.geometry.tolist(), crop=True, nodata=np.nan
            )

    out_profile = profile.copy()
    out_profile.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
    })
    return out_image[0], out_profile


def compute_slope(dem, pixel_size):
    """计算坡度 (度)"""
    dy, dx = np.gradient(dem, pixel_size)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_rad)


def compute_aspect(dem, pixel_size):
    """计算坡向 (度, 0-360)"""
    dy, dx = np.gradient(dem, pixel_size)
    aspect = np.degrees(np.arctan2(-dx, dy))
    aspect = np.where(aspect < 0, aspect + 360, aspect)
    slope = compute_slope(dem, pixel_size)
    aspect[slope == 0] = -1
    return aspect


def compute_twi(dem, pixel_size):
    """计算地形湿度指数 TWI（简化版：用坡度+局部统计近似，忽略汇流累积）"""
    dy, dx = np.gradient(dem, pixel_size)
    slope = np.sqrt(dy**2 + dx**2)
    slope_rad = np.arctan(slope)
    slope_rad = np.maximum(slope_rad, 1e-6)

    slope_factor = np.tan(slope_rad)
    slope_factor = np.clip(slope_factor, 0.01, 2)

    valid_mask = np.isfinite(dem)
    dem_filled = np.where(valid_mask, dem, 0)
    local_mean = uniform_filter(dem_filled, size=5, mode="constant", cval=0)

    twi = np.log(1.0 / slope_factor) + local_mean / 1000.0
    twi = np.where(valid_mask, twi, np.nan)
    twi = np.clip(twi, -5, 25).astype(np.float32)
    return twi


def compute_curvature(dem, pixel_size):
    """计算剖面曲率和平面曲率"""
    dy, dx = np.gradient(dem, pixel_size)
    dyy, dyx = np.gradient(dy, pixel_size)
    dxy, dxx = np.gradient(dx, pixel_size)

    denom = dx**2 + dy**2 + 1e-10

    pcurv = -(dxx * dx**2 + 2 * dxy * dx * dy + dyy * dy**2) / denom
    ccurv = -(dxx * dy**2 - 2 * dxy * dx * dy + dyy * dx**2) / denom

    pcurv = np.where(np.isfinite(pcurv), pcurv, 0)
    ccurv = np.where(np.isfinite(ccurv), ccurv, 0)

    pcurv = np.clip(pcurv, -0.1, 0.1).astype(np.float32)
    ccurv = np.clip(ccurv, -0.1, 0.1).astype(np.float32)

    return pcurv, ccurv


def compute_slope_position(dem, window_size=5):
    """计算坡位（相对高程位置：山脊/中坡/谷底）"""
    local_max = maximum_filter(dem, size=window_size, mode="constant", cval=0)
    local_min = minimum_filter(dem, size=window_size, mode="constant", cval=0)

    range_elev = local_max - local_min
    range_elev = np.maximum(range_elev, 1e-6)

    slope_pos = (dem - local_min) / range_elev
    slope_pos = np.clip(slope_pos, 0, 1).astype(np.float32)

    return slope_pos


def compute_roughness(dem, window_size=3):
    """计算粗糙度（局部高程标准差）"""
    valid_mask = np.isfinite(dem)
    dem_filled = np.where(valid_mask, dem, 0)

    local_mean = uniform_filter(dem_filled, size=window_size, mode="constant", cval=0)
    local_sqr_mean = uniform_filter(
        dem_filled**2, size=window_size, mode="constant", cval=0
    )
    variance = local_sqr_mean - local_mean**2
    variance = np.maximum(variance, 0)
    roughness = np.sqrt(variance)

    roughness = np.where(valid_mask, roughness, np.nan)
    roughness = np.clip(roughness, 0, 50).astype(np.float32)

    return roughness


def preprocess_dem(cfg: RegionConfig):
    """主处理流程：DEM预处理，输出8波段terrain_stack.tif"""
    print("=" * 60, flush=True)
    print("DEM 预处理 (输出8波段terrain_stack)", flush=True)
    print("=" * 60, flush=True)

    cfg.ensure_dirs()

    # 检查已存在（非force模式）
    if cfg.terrain_stack.exists() and not cfg.force:
        log(f"terrain_stack已存在，跳过: {cfg.terrain_stack}")
        return

    # 尝试复用同研究区其他年份的terrain_stack（地形不随年份变化）
    existing = cfg.find_existing_terrain_stack()
    if existing is not None and not cfg.force:
        log(f"复用已有terrain_stack: {existing}")
        shutil.copy2(existing, cfg.terrain_stack)
        log(f"已复制到: {cfg.terrain_stack}")
        return

    # 加载边界
    gdf = gpd.read_file(cfg.boundary)
    log(f"研究区: {cfg.region_name}, 目标CRS: {cfg.crs}")

    # 1. 重投影DEM
    log("\n[1/3] 重投影DEM...")
    t0 = time.time()
    dem_reprojected, dem_profile = reproject_dem(cfg.dem, cfg.crs, cfg.resolution, gdf)
    log(f"  耗时: {time.time()-t0:.1f}s")

    # 2. 裁剪到研究区
    log("\n[2/3] 裁剪DEM至研究区边界...")
    gdf_utm = gdf.to_crs(cfg.crs)
    dem_clipped, clip_profile = clip_with_boundary(dem_reprojected, dem_profile, gdf_utm)
    log(f"  裁剪后尺寸: {dem_clipped.shape}")
    del dem_reprojected

    # 3. 计算地形因子
    log("\n[3/3] 计算地形因子...")
    t0 = time.time()

    elevation = dem_clipped.astype(np.float32)
    slope = compute_slope(dem_clipped, cfg.resolution)
    aspect = compute_aspect(dem_clipped, cfg.resolution)
    twi = compute_twi(dem_clipped, cfg.resolution)
    pcurv, ccurv = compute_curvature(dem_clipped, cfg.resolution)
    slope_pos = compute_slope_position(dem_clipped)
    roughness = compute_roughness(dem_clipped)
    del dem_clipped

    log(f"  海拔: {np.nanmin(elevation):.1f} ~ {np.nanmax(elevation):.1f} m")
    log(f"  坡度: {np.nanmin(slope):.1f} ~ {np.nanmax(slope):.1f} deg")
    log(f"  坡向: {np.nanmin(aspect[aspect >= 0]):.1f} ~ {np.nanmax(aspect):.1f} deg")
    log(f"  TWI: {np.nanmin(twi):.2f} ~ {np.nanmax(twi):.2f}")
    log(f"  剖面曲率: {np.nanmin(pcurv):.4f} ~ {np.nanmax(pcurv):.4f}")
    log(f"  平面曲率: {np.nanmin(ccurv):.4f} ~ {np.nanmax(ccurv):.4f}")
    log(f"  坡位: {np.nanmin(slope_pos):.2f} ~ {np.nanmax(slope_pos):.2f}")
    log(f"  粗糙度: {np.nanmin(roughness):.2f} ~ {np.nanmax(roughness):.2f}")
    log(f"  耗时: {time.time()-t0:.1f}s")

    # 4. 堆叠为8波段TIF
    log("\n输出: " + str(cfg.terrain_stack))
    terrain_stack = np.stack(
        [elevation, slope, aspect, twi, pcurv, ccurv, slope_pos, roughness], axis=0
    )

    stack_profile = clip_profile.copy()
    stack_profile.update({
        "count": 8,
        "dtype": "float32",
        "nodata": np.nan,
        "compress": "lzw",
    })
    with rasterio.open(cfg.terrain_stack, "w", **stack_profile) as dst:
        dst.write(terrain_stack)
        for b, desc in enumerate(TERRAIN_DESCRIPTIONS, 1):
            dst.set_band_description(b, desc)

    print("\n" + "=" * 60, flush=True)
    print("DEM 预处理完成!")
    print(f"输出: {cfg.terrain_stack} (8波段: elevation, slope, aspect, twi, pcurv, ccurv, slope_pos, roughness)")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    cfg = parse_args()
    preprocess_dem(cfg)
