"""
Web数据预处理脚本
=================
将项目产出数据转换为Web可用的格式：
1. 分类TIF → WGS84 GeoTIFF + 彩色PNG
2. 统计信息预计算 (各等级面积、占比、变化)
3. 变化检测图 (年份间等级变化)
4. RF置信度图 (predict_proba)

用法:
    cd d:/jsjsjds/forest_carbon_tierin
    python -m web.scripts.generate_web_data
"""

import sys
from pathlib import Path

# 确保HierarchicalClassifier可以被pickle反序列化
import importlib.util
_hc_spec = importlib.util.spec_from_file_location(
    "hc_module",
    str(Path(__file__).resolve().parent.parent.parent / "src" / "06b_hierarchical_classifier.py")
)
_hc_mod = importlib.util.module_from_spec(_hc_spec)
_hc_spec.loader.exec_module(_hc_mod)
import __main__
__main__.HierarchicalClassifier = _hc_mod.HierarchicalClassifier
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask as rio_mask
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import box
import joblib
from collections import Counter

# 配置
REGIONS = ["ninger", "shuangbai"]
YEARS = [2019, 2023, 2024, 2025]
CRS_WGS84 = "EPSG:4326"

# 分类颜色 (RGBA): Low=橙红, Medium=琥珀, High=深绿
CLASS_COLORS = {
    1: (231, 76, 60, 220),    # Low - 红色
    2: (243, 156, 18, 220),   # Medium - 琥珀
    3: (39, 174, 96, 220),    # High - 绿色
}

# 输出目录
WEB_DATA_DIR = PROJECT_ROOT / "web" / "backend" / "data"


def ensure_dirs():
    """创建所有输出目录"""
    for region in REGIONS:
        for year in YEARS:
            (WEB_DATA_DIR / "maps" / region / str(year)).mkdir(parents=True, exist_ok=True)
        (WEB_DATA_DIR / "confidence" / region).mkdir(parents=True, exist_ok=True)
        (WEB_DATA_DIR / "change" / region).mkdir(parents=True, exist_ok=True)
    (WEB_DATA_DIR / "stats").mkdir(parents=True, exist_ok=True)
    (WEB_DATA_DIR / "boundaries").mkdir(parents=True, exist_ok=True)


def reproject_tif_to_wgs84(src_path: Path, dst_path: Path, region: str, dst_transform=None, dst_shape=None):
    """将TIF重投影到WGS84，并用研究区边界裁剪。
    可选: 传入dst_transform和dst_shape强制使用统一网格。
    """
    boundary_path = PROJECT_ROOT / f"data/{region}/raw/boundary/{region == 'ninger' and '宁洱县' or '双柏县'}_4326.geojson"
    
    with rasterio.open(src_path) as src:
        src_crs = src.crs
        src_data = src.read()
        
        # 如果已经是WGS84且不需要重采样
        if src_crs and src_crs.to_epsg() == 4326 and dst_transform is None:
            kwargs = src.meta.copy()
            kwargs['compress'] = 'lzw'
            with rasterio.open(dst_path, 'w', **kwargs) as dst:
                dst.write(src_data)
            return
        
        # 计算目标transform
        if dst_transform is not None and dst_shape is not None:
            # 使用统一网格
            transform = dst_transform
            height, width = dst_shape
        else:
            # 从源数据计算
            transform, width, height = calculate_default_transform(
                src_crs, CRS_WGS84, src.width, src.height, *src.bounds
            )
        
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': CRS_WGS84,
            'transform': transform,
            'width': width,
            'height': height,
            'compress': 'lzw'
        })
        
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=src_data[band-1],
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=CRS_WGS84,
                    resampling=Resampling.nearest
                )
    
    # 用边界裁剪
    if boundary_path.exists():
        gdf = gpd.read_file(boundary_path)
        # 稍微扩大裁剪范围
        gdf['geometry'] = gdf.buffer(0.005)
        
        with rasterio.open(dst_path) as src:
            out_image, out_transform = rio_mask(src, gdf.geometry, crop=True)
            out_meta = src.meta.copy()
        
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        with rasterio.open(dst_path, 'w', **out_meta) as dst:
            dst.write(out_image)


def tif_to_colored_png(tif_path: Path, png_path: Path, bounds_path: Path):
    """将分类TIF转为彩色PNG，保存bounds信息"""
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        bounds = src.bounds
    
    # 创建RGBA图像
    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    for cls_val, color in CLASS_COLORS.items():
        mask = data == cls_val
        for i, c in enumerate(color):
            rgba[:, :, i][mask] = c
    
    # 0值和nodata设为透明
    nodata_mask = (data == 0) | np.isnan(data) if np.issubdtype(data.dtype, np.floating) else (data == 0)
    rgba[nodata_mask, 3] = 0
    
    # 保存PNG
    from PIL import Image
    img = Image.fromarray(rgba, mode='RGBA')
    img.save(str(png_path))
    
    # 保存bounds
    bounds_info = {
        "south": bounds.bottom,
        "west": bounds.left,
        "north": bounds.top,
        "east": bounds.right
    }
    with open(bounds_path, 'w', encoding='utf-8') as f:
        json.dump(bounds_info, f, indent=2)
    
    return bounds_info


def compute_stats(region: str, year: int):
    """计算某区域某年份的统计信息"""
    tif_path = PROJECT_ROOT / f"output/{region}/{year}/classification/carbon_density_class.tif"
    
    if not tif_path.exists():
        return None
    
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        crs = src.crs
        # 正确计算像素面积
        if crs and crs.to_epsg() == 4326:
            # WGS84度->km: 1度约111km, 但面积与纬度相关
            # 使用中心纬度计算
            bounds = src.bounds
            lat = (bounds.top + bounds.bottom) / 2
            deg_to_km = 111.32 * np.cos(np.radians(lat))
            pixel_area = abs(src.transform[0] * src.transform[4]) * (deg_to_km ** 2)  # km2
        else:
            pixel_area = abs(src.transform[0] * src.transform[4]) / 1e6  # km2 (UTM)
    
    # 使用分类TIF本身的有效值，不再依赖forest_mask（尺寸可能不同）
    valid = data > 0
    
    data_valid = data[valid]
    
    # 各等级面积统计
    pixel_size_km2 = pixel_area
    class_stats = {}
    total_forest_area = 0
    
    for cls in [1, 2, 3]:
        count = int(np.sum(data_valid == cls))
        area = count * pixel_size_km2
        total_forest_area += area
        class_stats[f"Level{cls}"] = {
            "count": count,
            "area_km2": round(area, 2),
            "label": ["", "低碳密度", "中碳密度", "高碳密度"][cls]
        }
    
    # 计算占比
    for cls in [1, 2, 3]:
        key = f"Level{cls}"
        if total_forest_area > 0:
            class_stats[key]["percentage"] = round(
                class_stats[key]["area_km2"] / total_forest_area * 100, 2
            )
        else:
            class_stats[key]["percentage"] = 0
    
    return {
        "region": region,
        "year": year,
        "pixel_size_km2": round(pixel_size_km2, 6),
        "total_forest_area_km2": round(total_forest_area, 2),
        "total_pixels": int(np.sum(valid)),
        "class_stats": class_stats
    }


def compute_change(region: str, year1: int, year2: int):
    """计算两年间的碳汇等级变化（基于WGS84重投影后的数据）"""
    tif1_path = PROJECT_ROOT / f"output/{region}/{year1}/classification/carbon_density_class.tif"
    tif2_path = PROJECT_ROOT / f"output/{region}/{year2}/classification/carbon_density_class.tif"
    
    if not tif1_path.exists() or not tif2_path.exists():
        return None
    
    # 使用已重投影的WGS84 TIF进行变化分析（尺寸一致）
    wgs1 = WEB_DATA_DIR / "maps" / region / str(year1) / f"{region}_{year1}_wgs84.tif"
    wgs2 = WEB_DATA_DIR / "maps" / region / str(year2) / f"{region}_{year2}_wgs84.tif"
    
    if not wgs1.exists() or not wgs2.exists():
        print(f"  skip change {year1}->{year2}: wgs84 tif not ready")
        return None
    
    # 统一重采样到year1的WGS84网格
    with rasterio.open(wgs1) as src:
        data1 = src.read(1)
        profile = src.profile.copy()
        dst_transform = src.transform
        dst_crs = src.crs
        height, width = src.height, src.width
    
    # 将year2重采样到year1的网格
    data2 = np.zeros((height, width), dtype=src.dtypes[0])
    with rasterio.open(wgs2) as src2:
        reproject(
            source=rasterio.band(src2, 1),
            destination=data2,
            src_transform=src2.transform,
            src_crs=src2.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )
    
    valid = (data1 > 0) & (data2 > 0)
    d1 = data1[valid].astype(np.int16)
    d2 = data2[valid].astype(np.int16)
    
    # 变化: d2 - d1 (正=升级, 负=降级)
    change = d2 - d1
    
    total = len(d1)
    if total == 0:
        return None
    
    increased = int(np.sum(change > 0))
    decreased = int(np.sum(change < 0))
    unchanged = int(np.sum(change == 0))
    
    # 详细转移矩阵
    transition = {}
    for c1 in [1, 2, 3]:
        for c2 in [1, 2, 3]:
            count = int(np.sum((d1 == c1) & (d2 == c2)))
            if count > 0:
                transition[f"{c1}->{c2}"] = count
    
    # 生成变化图 (在WGS84网格上)
    change_data = np.zeros((height, width), dtype=np.int8)
    change_data[valid] = (data2[valid].astype(np.int16) - data1[valid].astype(np.int16)).astype(np.int8)
    
    # 保存变化图TIF
    out_path = WEB_DATA_DIR / "change" / region / f"change_{year1}_{year2}.tif"
    profile.update(dtype='int8', count=1, nodata=0)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(change_data, 1)
    
    # 生成变化PNG
    change_png_path = WEB_DATA_DIR / "change" / region / f"change_{year1}_{year2}.png"
    h, w = change_data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    upgrade_mask = change_data > 0
    rgba[upgrade_mask] = [46, 204, 113, 220]
    downgrade_mask = change_data < 0
    rgba[downgrade_mask] = [231, 76, 60, 220]
    unchanged_mask = (change_data == 0) & valid
    rgba[unchanged_mask] = [189, 195, 199, 150]
    
    from PIL import Image
    img = Image.fromarray(rgba, mode='RGBA')
    img.save(str(change_png_path))
    
    with rasterio.open(out_path) as src:
        bounds = src.bounds
    bounds_info = {
        "south": bounds.bottom, "west": bounds.left,
        "north": bounds.top, "east": bounds.right
    }
    with open(str(change_png_path).replace('.png', '_bounds.json'), 'w') as f:
        json.dump(bounds_info, f, indent=2)
    
    return {
        "region": region,
        "year1": year1,
        "year2": year2,
        "total_valid_pixels": total,
        "increased": increased,
        "decreased": decreased,
        "unchanged": unchanged,
        "increase_pct": round(increased / total * 100, 2) if total > 0 else 0,
        "decrease_pct": round(decreased / total * 100, 2) if total > 0 else 0,
        "unchanged_pct": round(unchanged / total * 100, 2) if total > 0 else 0,
        "transition_matrix": transition
    }


def generate_confidence_map(region: str):
    """基于模型的predict_proba生成置信度图"""
    # 优先使用HierarchicalClassifier，回退到RF
    hc_model_path = PROJECT_ROOT / f"output/{region}/2019/metrics/hc_model.pkl"
    rf_model_path = PROJECT_ROOT / f"output/{region}/2019/metrics/rf_model.pkl"
    
    model_path = hc_model_path if hc_model_path.exists() else rf_model_path
    feature_stack_path = PROJECT_ROOT / f"data/{region}/2019/aligned/feature_stack.tif"
    feature_names_path = PROJECT_ROOT / f"data/{region}/2019/aligned/feature_names.json"
    forest_mask_path = PROJECT_ROOT / f"data/{region}/2019/aligned/forest_mask.tif"
    
    if not model_path.exists() or not feature_stack_path.exists():
        print(f"  跳过置信度图: 缺少模型或特征数据")
        return
    
    # 加载模型和特征名
    model = joblib.load(model_path)
    with open(feature_names_path, 'r') as f:
        all_feature_names = json.load(f)["feature_names"]
    
    # 加载配置获取selected_features
    import yaml
    with open(PROJECT_ROOT / f"config/{region}.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    selected = cfg.get("selected_features", all_feature_names)
    # 空列表表示使用全部特征
    if not selected:
        selected = all_feature_names
    
    # 获取选中特征的索引
    selected_idx = [all_feature_names.index(name) for name in selected if name in all_feature_names]
    
    # 读取特征栈
    with rasterio.open(feature_stack_path) as src:
        features = src.read()
        profile = src.profile.copy()
        height, width = src.height, src.width
    
    # 读取森林掩膜
    with rasterio.open(forest_mask_path) as src:
        forest = src.read(1)
    
    # 展平
    n_bands, n_rows, n_cols = features.shape
    features_flat = features.reshape(n_bands, -1).T  # (pixels, bands)
    forest_flat = forest.flatten()
    
    # 仅预测森林区域
    valid = forest_flat == 1
    features_valid = features_flat[valid]
    
    # 选择特征
    features_selected = features_valid[:, selected_idx]
    
    # 替换NaN/Inf (允许30% NaN, 和训练时一致)
    nan_ratio = np.isnan(features_selected).sum(axis=1) / features_selected.shape[1]
    valid_features_mask = nan_ratio < 0.3
    if np.isnan(features_selected).any():
        col_means = np.nanmean(features_selected, axis=0)
        nan_mask = np.isnan(features_selected)
        for j in range(features_selected.shape[1]):
            features_selected[nan_mask[:, j], j] = col_means[j] if np.isfinite(col_means[j]) else 0
    
    # 仅预测通过NaN过滤的森林区域
    features_selected = features_selected[valid_features_mask]
    valid_indices = np.where(valid)[0]
    valid_indices = valid_indices[valid_features_mask]
    
    # 分批预测概率
    print(f"  预测置信度: {len(features_selected)} 像素...")
    batch_size = 100000
    probas = []
    for i in range(0, len(features_selected), batch_size):
        batch = features_selected[i:i+batch_size]
        probas.append(model.predict_proba(batch))
    
    all_probas = np.vstack(probas)
    confidence = np.max(all_probas, axis=1)  # 最大概率作为置信度
    
    # 写回完整栅格
    confidence_full = np.zeros(n_rows * n_cols, dtype=np.float32)
    confidence_full[valid_indices] = confidence
    confidence_full = confidence_full.reshape(n_rows, n_cols)
    
    # 保存为TIF (原始CRS)
    out_path = WEB_DATA_DIR / "confidence" / region / f"confidence_{region}_2019.tif"
    profile.update(dtype='float32', count=1, nodata=0)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(confidence_full, 1)
    
    # 重投影到WGS84并生成PNG
    wgs84_path = WEB_DATA_DIR / "confidence" / region / f"confidence_{region}_2019_wgs84.tif"
    with rasterio.open(out_path) as src:
        transform, w, h = calculate_default_transform(
            src.crs, CRS_WGS84, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({'crs': CRS_WGS84, 'transform': transform, 'width': w, 'height': h, 'compress': 'lzw'})
        with rasterio.open(wgs84_path, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=CRS_WGS84,
                resampling=Resampling.bilinear
            )
    
    # 生成置信度PNG
    with rasterio.open(wgs84_path) as src:
        conf_data = src.read(1)
        bounds = src.bounds
    
    h, w = conf_data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # 高置信度 (>0.8) = 绿色, 中 (0.6-0.8) = 黄色, 低 (<0.6) = 红色
    high = (conf_data >= 0.8) & (conf_data > 0)
    mid = (conf_data >= 0.6) & (conf_data < 0.8) & (conf_data > 0)
    low = (conf_data > 0) & (conf_data < 0.6)
    
    rgba[high] = [39, 174, 96, 200]    # 绿
    rgba[mid] = [243, 156, 18, 200]    # 黄
    rgba[low] = [231, 76, 60, 200]     # 红
    
    from PIL import Image
    img = Image.fromarray(rgba, mode='RGBA')
    png_path = WEB_DATA_DIR / "confidence" / region / f"confidence_{region}_2019.png"
    img.save(str(png_path))
    
    bounds_info = {
        "south": bounds.bottom, "west": bounds.left,
        "north": bounds.top, "east": bounds.right
    }
    with open(str(png_path).replace('.png', '_bounds.json'), 'w') as f:
        json.dump(bounds_info, f, indent=2)
    
    # 置信度统计
    conf_valid = conf_data[conf_data > 0]
    conf_stats = {
        "region": region,
        "high_pct": round(np.sum(conf_valid >= 0.8) / len(conf_valid) * 100, 2) if len(conf_valid) > 0 else 0,
        "mid_pct": round(np.sum((conf_valid >= 0.6) & (conf_valid < 0.8)) / len(conf_valid) * 100, 2) if len(conf_valid) > 0 else 0,
        "low_pct": round(np.sum(conf_valid < 0.6) / len(conf_valid) * 100, 2) if len(conf_valid) > 0 else 0,
        "mean_confidence": round(float(np.mean(conf_valid)), 4) if len(conf_valid) > 0 else 0,
        "median_confidence": round(float(np.median(conf_valid)), 4) if len(conf_valid) > 0 else 0
    }
    
    stats_path = WEB_DATA_DIR / "confidence" / region / f"confidence_stats_{region}.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(conf_stats, f, indent=2, ensure_ascii=False)
    
    print(f"  置信度统计: 高={conf_stats['high_pct']}%, 中={conf_stats['mid_pct']}%, 低={conf_stats['low_pct']}%")


def copy_boundaries():
    """复制WGS84边界GeoJSON到Web数据目录"""
    for region in REGIONS:
        name = "宁洱县" if region == "ninger" else "双柏县"
        src = PROJECT_ROOT / f"data/{region}/raw/boundary/{name}_4326.geojson"
        dst = WEB_DATA_DIR / "boundaries" / f"{region}.geojson"
        if src.exists():
            import shutil
            shutil.copy2(str(src), str(dst))


def main():
    print("=" * 60)
    print("Web数据预处理")
    print("=" * 60)
    
    ensure_dirs()
    
    # 1. 重投影分类TIF + 生成PNG
    print("\n[1/5] 重投影分类TIF并生成PNG...")
    for region in REGIONS:
        # 先计算该区域统一的WGS84网格（基于UTM的2024年数据）
        unified_transform = None
        unified_shape = None
        for year in YEARS:
            src_tif = PROJECT_ROOT / f"output/{region}/{year}/classification/carbon_density_class.tif"
            if not src_tif.exists():
                continue
            with rasterio.open(src_tif) as src:
                if src.crs and src.crs.to_epsg() != 4326:
                    t, w, h = calculate_default_transform(
                        src.crs, CRS_WGS84, src.width, src.height, *src.bounds
                    )
                    unified_transform = t
                    unified_shape = (h, w)
                    print(f"  {region}: unified grid {w}x{h}")
                    break
        
        for year in YEARS:
            src_tif = PROJECT_ROOT / f"output/{region}/{year}/classification/carbon_density_class.tif"
            if not src_tif.exists():
                print(f"  skip {region}/{year}: file not found")
                continue
            
            wgs84_tif = WEB_DATA_DIR / "maps" / region / str(year) / f"{region}_{year}_wgs84.tif"
            png_path = WEB_DATA_DIR / "maps" / region / str(year) / f"{region}_{year}.png"
            bounds_path = WEB_DATA_DIR / "maps" / region / str(year) / f"{region}_{year}_bounds.json"
            
            if wgs84_tif.exists() and png_path.exists():
                print(f"  {region}/{year}: exists, skip")
                continue
            
            print(f"  processing {region}/{year}...")
            try:
                reproject_tif_to_wgs84(src_tif, wgs84_tif, region, 
                                       dst_transform=unified_transform, 
                                       dst_shape=unified_shape)
                tif_to_colored_png(wgs84_tif, png_path, bounds_path)
                print(f"  {region}/{year}: done")
            except Exception as e:
                print(f"  {region}/{year}: error - {e}")
    
    # 2. 预计算统计
    print("\n[2/5] 预计算统计信息...")
    all_stats = {}
    for region in REGIONS:
        region_stats = []
        for year in YEARS:
            stats = compute_stats(region, year)
            if stats:
                region_stats.append(stats)
                print(f"  {region}/{year}: forest_area={stats['total_forest_area_km2']}km2")
        all_stats[region] = region_stats
    
    # 保存统计JSON
    stats_path = WEB_DATA_DIR / "stats" / "all_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    # 3. 变化检测
    print("\n[3/5] 计算变化检测...")
    change_pairs = [(2019, 2023), (2023, 2024), (2024, 2025), (2019, 2025)]
    all_changes = {}
    for region in REGIONS:
        region_changes = []
        for y1, y2 in change_pairs:
            change = compute_change(region, y1, y2)
            if change:
                region_changes.append(change)
                print(f"  {region} {y1}->{y2}: 增加={change['increase_pct']}%, 减少={change['decrease_pct']}%")
        all_changes[region] = region_changes
    
    change_path = WEB_DATA_DIR / "stats" / "all_changes.json"
    with open(change_path, 'w', encoding='utf-8') as f:
        json.dump(all_changes, f, indent=2, ensure_ascii=False)
    
    # 4. 置信度图
    print("\n[4/5] 生成置信度图...")
    for region in REGIONS:
        print(f"  处理 {region}...")
        try:
            generate_confidence_map(region)
        except Exception as e:
            print(f"  {region} 置信度图生成失败: {e}")
    
    # 5. 复制边界
    print("\n[5/5] 复制边界数据...")
    copy_boundaries()
    
    # 复制模型指标
    for region in REGIONS:
        for src_name, dst_name in [
            (f"output/{region}/2019/metrics/model_metrics.json", f"stats/model_metrics_{region}.json"),
            (f"output/{region}/2019/metrics/feature_importance.csv", f"stats/feature_importance_{region}.csv"),
            (f"output/{region}/2019/classification/classification_breaks.json", f"stats/classification_breaks_{region}_2019.json"),
            (f"data/{region}/2019/aligned/jenks_breaks.json", f"stats/jenks_breaks_{region}.json"),
        ]:
            src = PROJECT_ROOT / src_name
            dst = WEB_DATA_DIR / dst_name
            if src.exists():
                import shutil
                shutil.copy2(str(src), str(dst))
    
    print("\n" + "=" * 60)
    print("预处理完成!")
    print(f"输出目录: {WEB_DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
