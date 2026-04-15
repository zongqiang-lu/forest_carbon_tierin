"""
植被指数计算脚本
================
功能：从landsat_sr_stack.tif(7波段)读取Landsat波段，计算13种光谱衍生特征，
      输出indices_stack.tif(13波段)

基础植被指数：
- NDVI = (B5-B4)/(B5+B4)  (Rouse et al., 1974)
- EVI = 2.5×(B5-B4)/(B5+6×B4-7.5×B2+1)  (Huete et al., 2002)
- NDMI = (B5-B6)/(B5+B6)  (Gao, 1996)
- SAVI = 1.5×(B5-B4)/(B5+B4+0.5)  (Huete, 1988)
- NDWI = (B3-B5)/(B3+B5)  (McFeeters, 1996)
- MSAVI = (2*B5 + 1 - sqrt((2*B5 + 1)^2 - 8*(B5 - B4)))/2  (Qi et al., 1994)
- NBR = (B5-B7)/(B5+B7)  (Key et al., 1998)

波段比值特征：
- SR_B5B4 = B5/B4 (近红外/红光)
- SR_B6B5 = B6/B5 (短波红外1/近红外)
- SR_B7B5 = B7/B5 (短波红外2/近红外)

缨帽变换分量（Landsat 8 OLI系数）：
- TCB = 0.3029*B1 + 0.2786*B2 + 0.4733*B3 + 0.5599*B4 + 0.508*B5 + 0.1872*B6 + 0.1029*B7 (亮度)
- TCG = -0.2941*B1 - 0.243*B2 - 0.5424*B3 + 0.7276*B4 + 0.0713*B5 - 0.1608*B6 - 0.0207*B7 (绿度)
- TCW = 0.1511*B1 + 0.1973*B2 + 0.3283*B3 + 0.3407*B4 - 0.7117*B5 - 0.4559*B6 - 0.2623*B7 (湿度)

波段映射 (landsat_sr_stack.tif):
  Band 1 = B1 (Coastal/Aerosol)
  Band 2 = B2 (Blue)
  Band 3 = B3 (Green)
  Band 4 = B4 (Red)
  Band 5 = B5 (NIR)
  Band 6 = B6 (SWIR1)
  Band 7 = B7 (SWIR2)
"""

import sys
import time
import numpy as np
import rasterio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import parse_args, RegionConfig

INDEX_NAMES = [
    "NDVI",
    "EVI",
    "NDMI",
    "SAVI",
    "NDWI",
    "MSAVI",
    "NBR",
    "SR_B5B4",
    "SR_B6B5",
    "SR_B7B5",
    "TCB",
    "TCG",
    "TCW",
]
INDEX_DESCRIPTIONS = [
    "Normalized Difference Vegetation Index",
    "Enhanced Vegetation Index",
    "Normalized Difference Moisture Index",
    "Soil Adjusted Vegetation Index",
    "Normalized Difference Water Index",
    "Modified Soil Adjusted Vegetation Index",
    "Normalized Burn Ratio",
    "Simple Ratio NIR/Red",
    "Simple Ratio SWIR1/NIR",
    "Simple Ratio SWIR2/NIR",
    "Tasseled Cap Brightness",
    "Tasseled Cap Greenness",
    "Tasseled Cap Wetness",
]


def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def safe_divide(numerator, denominator, fill_value=np.nan):
    """安全除法，避免除零"""
    result = np.full_like(numerator, fill_value, dtype=np.float32)
    valid = np.isfinite(denominator) & (denominator != 0)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def compute_indices(bands: dict) -> dict:
    """计算所有植被指数、波段比值和缨帽变换特征"""
    indices = {}
    # 基础植被指数
    indices["NDVI"] = safe_divide(bands["B5"] - bands["B4"], bands["B5"] + bands["B4"])
    indices["EVI"] = safe_divide(
        2.5 * (bands["B5"] - bands["B4"]),
        bands["B5"] + 6.0 * bands["B4"] - 7.5 * bands["B2"] + 1.0,
    )
    indices["NDMI"] = safe_divide(bands["B5"] - bands["B6"], bands["B5"] + bands["B6"])
    indices["SAVI"] = safe_divide(
        1.5 * (bands["B5"] - bands["B4"]), bands["B5"] + bands["B4"] + 0.5
    )
    indices["NDWI"] = safe_divide(bands["B3"] - bands["B5"], bands["B3"] + bands["B5"])

    # 新增植被指数
    # MSAVI计算
    msavi_term = (2 * bands["B5"] + 1) ** 2 - 8 * (bands["B5"] - bands["B4"])
    msavi_term = np.where(msavi_term < 0, 0, msavi_term)  # 避免负数开根号
    indices["MSAVI"] = (2 * bands["B5"] + 1 - np.sqrt(msavi_term)) / 2
    # NBR计算
    indices["NBR"] = safe_divide(bands["B5"] - bands["B7"], bands["B5"] + bands["B7"])

    # 波段比值特征
    indices["SR_B5B4"] = safe_divide(bands["B5"], bands["B4"])
    indices["SR_B6B5"] = safe_divide(bands["B6"], bands["B5"])
    indices["SR_B7B5"] = safe_divide(bands["B7"], bands["B5"])

    # 缨帽变换分量（Landsat 8 OLI系数）
    indices["TCB"] = (
        0.3029 * bands["B1"]
        + 0.2786 * bands["B2"]
        + 0.4733 * bands["B3"]
        + 0.5599 * bands["B4"]
        + 0.508 * bands["B5"]
        + 0.1872 * bands["B6"]
        + 0.1029 * bands["B7"]
    )
    indices["TCG"] = (
        -0.2941 * bands["B1"]
        - 0.243 * bands["B2"]
        - 0.5424 * bands["B3"]
        + 0.7276 * bands["B4"]
        + 0.0713 * bands["B5"]
        - 0.1608 * bands["B6"]
        - 0.0207 * bands["B7"]
    )
    indices["TCW"] = (
        0.1511 * bands["B1"]
        + 0.1973 * bands["B2"]
        + 0.3283 * bands["B3"]
        + 0.3407 * bands["B4"]
        - 0.7117 * bands["B5"]
        - 0.4559 * bands["B6"]
        - 0.2623 * bands["B7"]
    )

    return indices


def compute_vegetation_indices(cfg: RegionConfig):
    """主处理流程：计算植被指数"""
    print("=" * 60, flush=True)
    print("植被指数计算 (输出13波段indices_stack)", flush=True)
    print("=" * 60, flush=True)

    cfg.ensure_dirs()

    # 增量执行：已存在则跳过
    if cfg.indices_stack.exists() and not cfg.force:
        log(f"植被指数栈已存在，跳过: {cfg.indices_stack}")
        return

    # 读取landsat_sr_stack.tif
    log(f"[1/2] 读取Landsat SR数据: {cfg.landsat_sr_stack}")
    t0 = time.time()

    with rasterio.open(cfg.landsat_sr_stack) as src:
        n_bands = src.count
        profile = src.profile.copy()
        # 读取7个波段
        band_data = src.read()  # (7, h, w)
        log(f"  波段数: {n_bands}, 尺寸: {band_data.shape[1]}x{band_data.shape[2]}")

    # 波段映射: Band1=B1, Band2=B2, ..., Band7=B7
    band_map = {
        "B1": band_data[0], "B2": band_data[1], "B3": band_data[2],
        "B4": band_data[3], "B5": band_data[4], "B6": band_data[5],
        "B7": band_data[6],
    }
    del band_data
    log(f"  读取耗时: {time.time()-t0:.1f}s")

    # 计算植被指数
    log("\n[2/2] 计算植被指数...")
    indices = compute_indices(band_map)
    del band_map

    for name in INDEX_NAMES:
        data = indices[name]
        valid = data[np.isfinite(data)]
        if len(valid) > 0:
            log(f"  {name}: min={valid.min():.4f}, max={valid.max():.4f}, "
                f"mean={valid.mean():.4f}")

    # 堆叠为13波段TIF
    log(f"\n输出: {cfg.indices_stack}")
    indices_stack = np.stack([indices[name] for name in INDEX_NAMES], axis=0)

    stack_profile = profile.copy()
    stack_profile.update({
        "count": len(INDEX_NAMES),
        "dtype": "float32",
        "nodata": np.nan,
        "compress": "lzw",
    })
    with rasterio.open(cfg.indices_stack, "w", **stack_profile) as dst:
        dst.write(indices_stack)
        for b, desc in enumerate(INDEX_DESCRIPTIONS, 1):
            dst.set_band_description(b, desc)

    print("\n" + "=" * 60, flush=True)
    print("植被指数计算完成!")
    print(f"输出: {cfg.indices_stack} (13波段: {', '.join(INDEX_NAMES)})")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    cfg = parse_args()
    compute_vegetation_indices(cfg)
