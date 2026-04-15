"""
AGB分级反演与制图脚本
======================
功能：全区域分级分类预测、分级专题图、混淆矩阵图、特征重要性图
支持跨年份模型预测：非基准年使用基准年训练的模型
支持模型类型：RandomForestClassifier / HierarchicalClassifier
"""

import sys
import time
import numpy as np
import rasterio
import json
import joblib
from pathlib import Path

# 确保HierarchicalClassifier可以被pickle反序列化
# 模型保存时HierarchicalClassifier的__module__是__main__，需要注入到当前__main__
import importlib.util
_hc_spec = importlib.util.spec_from_file_location(
    "hc_module",
    str(Path(__file__).resolve().parent / "06b_hierarchical_classifier.py")
)
_hc_mod = importlib.util.module_from_spec(_hc_spec)
_hc_spec.loader.exec_module(_hc_mod)
import __main__
__main__.HierarchicalClassifier = _hc_mod.HierarchicalClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import pandas as pd
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import parse_args, RegionConfig

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CLASS_COLORS = {
    3: ["#ffffcc", "#41b6c4", "#253494"],
    5: ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"],
}
CLASS_LABELS_EN = {
    3: ["Low", "Medium", "High"],
    5: ["Low", "Rel.Low", "Medium", "Rel.High", "High"],
}


def log(msg):
    t = time.strftime("%H:%M:%S")
    try:
        print(f"[{t}] {msg}", flush=True)
    except UnicodeEncodeError:
        print(f"[{t}] {msg.encode('ascii', 'replace').decode('ascii')}", flush=True)


def predict_classification_map(cfg: RegionConfig):
    log("=" * 60)
    log(f"AGB分级反演与制图 ({cfg.region_name} {cfg.year})")
    log("=" * 60)

    cfg.ensure_dirs()

    # 1. 加载模型（支持跨年份）
    model_path = cfg.get_model_path()
    log(f"[1/4] 加载模型: {model_path}")
    if not model_path.exists():
        log(f"错误: 模型文件不存在: {model_path}")
        log("请先运行完整流程(01-06)训练模型")
        return
    t0 = time.time()
    rf_model = joblib.load(model_path)
    log(f"  模型加载完成, 耗时: {time.time()-t0:.1f}s")

    # 检测模型类型
    from importlib import import_module
    model_type_name = type(rf_model).__name__
    is_hierarchical = model_type_name == "HierarchicalClassifier"
    if is_hierarchical:
        log(f"  模型类型: HierarchicalClassifier (分层二分类)")
    else:
        log(f"  模型类型: {model_type_name}")

    with rasterio.open(cfg.valid_mask) as src:
        valid_mask = src.read(1) == 1
        mask_profile = src.profile.copy()
        height, width = valid_mask.shape

    with rasterio.open(cfg.feature_stack) as src:
        n_bands_in_stack = src.count
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform
        log(f"  特征栈: {n_bands_in_stack} bands, {height}x{width}")

    # 读取feature_names.json获取波段名称（从当前年份的特征栈目录）
    with open(cfg.feature_names_json) as f:
        all_feature_names = json.load(f).get("feature_names", [])

    # 确定需要读取的波段索引（模型可能只用部分特征）
    if hasattr(rf_model, 'n_features_in_'):
        model_n_features = rf_model.n_features_in_
    elif is_hierarchical and hasattr(rf_model, 'clf_1_vs_23'):
        model_n_features = rf_model.clf_1_vs_23.n_features_in_
    else:
        model_n_features = n_bands_in_stack

    selected_features = cfg.selected_features

    if selected_features and len(selected_features) == model_n_features:
        # 模型使用了特征筛选，只读取对应波段
        band_indices = [all_feature_names.index(f) + 1 for f in selected_features
                        if f in all_feature_names]  # 1-based
        log(f"  特征筛选: 读取{len(band_indices)}/{n_bands_in_stack}波段")
    elif model_n_features < n_bands_in_stack:
        # 回退：读取前N个波段
        band_indices = list(range(1, model_n_features + 1))
        log(f"  读取前{model_n_features}波段")
    else:
        band_indices = list(range(1, n_bands_in_stack + 1))

    # 提取有效像元特征
    log("  提取有效像元特征...")
    valid_indices = np.where(valid_mask.flatten())[0]
    n_valid = len(valid_indices)
    log(f"  有效像元: {n_valid}")

    # 一次性读取需要的波段
    with rasterio.open(cfg.feature_stack) as src:
        stack = src.read(band_indices)  # (n_selected_bands, h, w)
    n_features = len(band_indices)
    stack_flat = stack.reshape(n_features, -1).T  # (h*w, n_features)
    X_valid = stack_flat[valid_indices].astype(np.float32)
    del stack, stack_flat

    log(f"  特征提取完成, 耗时: {time.time()-t0:.1f}s")

    # 2. 批量预测分类
    log("[2/4] 全区域分类预测...")
    t0 = time.time()

    # 允许最多30%的NaN特征
    nan_ratio = np.isnan(X_valid).sum(axis=1) / X_valid.shape[1]
    finite_mask = nan_ratio < 0.3
    X_predict = X_valid[finite_mask]
    predict_indices = valid_indices[finite_mask]
    del X_valid

    # 对NaN特征用列均值填充（HierarchicalClassifier和RF都需要有限值）
    if np.isnan(X_predict).any():
        col_means = np.nanmean(X_predict, axis=0)
        nan_mask = np.isnan(X_predict)
        for j in range(X_predict.shape[1]):
            X_predict[nan_mask[:, j], j] = col_means[j] if np.isfinite(col_means[j]) else 0

    class_pred_flat = np.full(height * width, 0, dtype=np.uint8)
    batch_size = 200000
    for start in range(0, len(X_predict), batch_size):
        end = min(start + batch_size, len(X_predict))
        pred = rf_model.predict(X_predict[start:end]).astype(np.uint8)
        class_pred_flat[predict_indices[start:end]] = pred
        log(f"    预测进度: {end}/{len(X_predict)}")

    class_pred = class_pred_flat.reshape(height, width)
    del class_pred_flat, X_predict

    # 保存分类结果
    out_profile = profile.copy()
    out_profile.update({"count": 1, "dtype": "uint8", "nodata": 0, "compress": "lzw"})

    with rasterio.open(cfg.classification_tif, "w", **out_profile) as dst:
        dst.write(class_pred, 1)

    for i in range(cfg.n_classes):
        n_px = (class_pred == i + 1).sum()
        area_km2 = n_px * cfg.resolution * cfg.resolution / 1e6
        log(f"  级别{i+1}: {n_px}像元, {area_km2:.2f} km2")

    log(f"  分类预测完成, 耗时: {time.time()-t0:.1f}s")

    # 3. 读取断点信息（从模型来源年份）
    log("[3/4] 读取Jenks断点信息...")
    breaks = []
    jenks_path = cfg.get_jenks_breaks_path()
    if jenks_path.exists():
        with open(jenks_path, encoding="utf-8") as f:
            breaks_info = json.load(f)
        breaks = breaks_info.get("breaks", [])
        for i in range(len(breaks) - 1):
            log(f"  级别{i+1}: {breaks[i]:.2f} ~ {breaks[i+1]:.2f} t/ha")
        log(f"  (断点来自模型训练年份: {cfg.model_year})")
    else:
        # 回退到当前年份的jenks文件
        if cfg.jenks_breaks_json.exists():
            with open(cfg.jenks_breaks_json, encoding="utf-8") as f:
                breaks_info = json.load(f)
            breaks = breaks_info.get("breaks", [])
        else:
            breaks_info = {}

    # 保存分类断点信息
    model_type_str = "HierarchicalClassifier" if is_hierarchical else "RF_Classifier"
    class_breaks_info = {
        "source": f"{model_type_str}_predicted",
        "method": "Jenks Natural Breaks (from AGB reference)",
        "n_classes": cfg.n_classes,
        "region": cfg.region_id,
        "year": cfg.year,
        "model_source": f"{cfg.model_region_id}_{cfg.model_year}",
        "model_type": model_type_str,
        "breaks": [float(b) for b in breaks],
        "class_labels": CLASS_LABELS_EN.get(cfg.n_classes, [f"L{i+1}" for i in range(cfg.n_classes)]),
        "class_ranges": [f"{breaks[i]:.2f}~{breaks[i+1]:.2f}" for i in range(len(breaks)-1)] if len(breaks) > 1 else [],
    }
    with open(cfg.classification_breaks_json, "w", encoding="utf-8") as f:
        json.dump(class_breaks_info, f, ensure_ascii=False, indent=2)

    # 4. 可视化
    log("[4/4] 生成可视化图表...")
    generate_figures(cfg, class_pred, breaks, profile, crs, transform, is_hierarchical)

    log("=" * 60)
    log("AGB分级反演与制图完成!")
    log("=" * 60)


def generate_figures(cfg, class_pred, breaks, profile, crs, transform, is_hierarchical=False):
    """生成所有可视化图表"""

    left = transform.c
    top = transform.f
    right = left + profile["width"] * transform.a
    bottom = top + profile["height"] * transform.e
    extent = [left, right, bottom, top]

    n_classes = cfg.n_classes

    # --- 1. 树智碳汇分级专题图 ---
    log("  生成树智碳汇分级专题图...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    colors = CLASS_COLORS.get(n_classes, CLASS_COLORS[5][:n_classes])
    cmap = mcolors.ListedColormap(["white"] + colors)
    bounds = [i - 0.5 for i in range(n_classes + 2)]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(class_pred, cmap=cmap, norm=norm, extent=extent)

    try:
        gdf = gpd.read_file(cfg.boundary)
        if gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.5)
    except Exception as e:
        log(f"    边界绘制跳过: {e}")

    class_labels_cn = {
        3: ["低碳密度", "中碳密度", "高碳密度"],
        5: ["低碳密度", "较低碳密度", "中碳密度", "较高碳密度", "高碳密度"],
    }.get(n_classes, [f"级别{i+1}" for i in range(n_classes)])
    if len(breaks) >= n_classes + 1:
        legend_elements = [
            Patch(facecolor=colors[i], edgecolor="black",
                  label=f"{class_labels_cn[i]} ({breaks[i]:.1f}-{breaks[i+1]:.1f})")
            for i in range(n_classes)
        ]
    else:
        legend_elements = [
            Patch(facecolor=colors[i], edgecolor="black",
                  label=class_labels_cn[i])
            for i in range(n_classes)
        ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              title="Carbon density (t/ha)", title_fontsize=10)
    ax.set_title(f"{cfg.region_name} {cfg.year} Forest Carbon Density Classification",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    plt.tight_layout()
    fig.savefig(cfg.figures_dir / "carbon_density_classification.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- 2. 混淆矩阵图 (仅当有验证数据时) ---
    log("  检查混淆矩阵数据...")
    if cfg.validation_data.exists():
        val = np.load(cfg.validation_data)
        y_val = val["y_val"]
        y_val_pred = val["y_val_pred"]

        from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
        cm = confusion_matrix(y_val, y_val_pred, labels=list(range(1, n_classes + 1)))
        oa = accuracy_score(y_val, y_val_pred)
        kappa = cohen_kappa_score(y_val, y_val_pred)

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax, shrink=0.8)

        class_names = [f"L{i+1}" for i in range(n_classes)]
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Reference", fontsize=12)
        ax.set_title(f"Confusion Matrix (OA={oa:.4f}, Kappa={kappa:.4f})",
                     fontsize=13, fontweight="bold")

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=11)

        plt.tight_layout()
        fig.savefig(cfg.figures_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
        plt.close()
    else:
        log("  无验证数据（预测模式），跳过混淆矩阵图")

    # --- 3. 特征重要性图 (仅当模型有特征重要性时) ---
    log("  检查特征重要性数据...")
    model_metrics_path = cfg.get_model_path().parent / "feature_importance.csv"
    if model_metrics_path.exists():
        fi = pd.read_csv(model_metrics_path).sort_values("importance", ascending=True)
        fig, ax = plt.subplots(1, 1, figsize=(10, max(6, len(fi) * 0.3)))
        bars = ax.barh(fi["feature"], fi["importance"], color="steelblue", edgecolor="black")
        for bar, val in zip(bars, fi["importance"]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)
        ax.set_xlabel("Gini Importance", fontsize=12)
        model_label = "HierarchicalClassifier" if is_hierarchical else "Random Forest"
        ax.set_title(f"{model_label} Feature Importance ({cfg.model_region_id}_{cfg.model_year})",
                     fontsize=13, fontweight="bold")
        ax.set_xlim(0, fi["importance"].max() * 1.2)
        plt.tight_layout()
        fig.savefig(cfg.figures_dir / "feature_importance.png", dpi=200, bbox_inches="tight")
        plt.close()
    else:
        log("  无特征重要性数据，跳过")

    # --- 4. 精度指标表 (仅当有模型指标时) ---
    log("  检查模型指标数据...")
    model_metrics_json_path = cfg.get_model_path().parent / "model_metrics.json"
    if model_metrics_json_path.exists():
        with open(model_metrics_json_path, encoding="utf-8") as f:
            metrics = json.load(f)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis("off")

        cv_label = f"{metrics.get('cv_folds', '5')}-fold CV OA"
        model_display = metrics.get('model_type', 'HierarchicalClassifier' if is_hierarchical else 'RandomForestClassifier')
        table_data = [
            ["Metric", "Value"],
            ["Model", model_display],
            ["Region", cfg.region_name],
            ["Year", str(cfg.year)],
            ["Model Source", f"{cfg.model_region_id}_{cfg.model_year}"],
        ]
        # CV OA仅对RF模型可用
        if 'cv_oa_mean' in metrics:
            table_data.append([cv_label, f"{metrics['cv_oa_mean']:.4f} +/- {metrics['cv_oa_std']:.4f}"])
        table_data.extend([
            ["Validation OA", f"{metrics['val_oa']:.4f}"],
            ["Validation Kappa", f"{metrics['val_kappa']:.4f}"],
        ])

        pa = metrics.get("pa_per_class", {})
        ua = metrics.get("ua_per_class", {})
        for i in range(n_classes):
            key = f"Level{i+1}"
            pa_val = pa.get(key, 0)
            ua_val = ua.get(key, 0)
            table_data.append([f"PA Level{i+1}", f"{pa_val:.4f}"])
            table_data.append([f"UA Level{i+1}", f"{ua_val:.4f}"])

        table_data.extend([
            ["Train samples", f"{metrics['n_train']}"],
            ["Val samples", f"{metrics['n_val']}"],
            ["Features", f"{metrics['n_features']}"],
        ])

        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.4)
        ax.set_title(f"{model_display} Validation Metrics", fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()
        fig.savefig(cfg.figures_dir / "metrics_summary.png", dpi=200, bbox_inches="tight")
        plt.close()
    else:
        log("  无模型指标数据，跳过")

    log("  所有图表生成完成!")


if __name__ == "__main__":
    cfg = parse_args()
    predict_classification_map(cfg)
