"""
样本提取与RF分类建模脚本
========================
功能：像元级特征-标签配对、质量筛选(坡度+NDVI+AGB+质量掩膜)、
      分层等量采样(宁缺毋滥)、空间去自相关、
      随机森林分类器(5级Jenks分级标签)、10折CV + 30%独立验证、特征重要性

标签：使用05_label_preparation.py输出的Jenks 5级分级结果
评估指标：OA, Kappa, 各级PA/UA, 混淆矩阵
"""

import sys
import time
import numpy as np
import rasterio
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, classification_report,
    confusion_matrix
)
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import parse_args, RegionConfig


def log(msg):
    t = time.strftime("%H:%M:%S")
    try:
        print(f"[{t}] {msg}", flush=True)
    except UnicodeEncodeError:
        print(f"[{t}] {msg.encode('ascii', 'replace').decode('ascii')}", flush=True)


def load_feature_stack(cfg: RegionConfig):
    """加载特征栈和AGB分级标签，支持特征筛选"""
    with rasterio.open(cfg.feature_stack) as src:
        stack = src.read()
    with open(cfg.feature_names_json) as f:
        meta = json.load(f)
    with rasterio.open(cfg.agb_class) as src:
        agb_class = src.read(1)
    with rasterio.open(cfg.valid_mask) as src:
        valid_mask = src.read(1) == 1

    all_features = meta["feature_names"]

    # 特征筛选：如果YAML中配置了selected_features，只保留这些
    if cfg.selected_features:
        selected_indices = [all_features.index(f) for f in cfg.selected_features if f in all_features]
        stack = stack[selected_indices]
        all_features = [all_features[i] for i in selected_indices]
        log(f"  特征筛选: {meta['feature_names']} -> {all_features} ({len(all_features)}/{meta['n_features']})")

    return stack, all_features, agb_class, valid_mask


def stratified_spatial_sampling(valid_mask, agb_class, n_classes,
                                 min_spacing=3, per_class_target=0,
                                 fraction=0.15, seed=42):
    """
    分层等量空间采样（宁缺毋滥）
    
    策略：
    - per_class_target > 0: 每级取等量样本，避免低等级主导模型
    - per_class_target = 0: 按fraction比例采样（原始逻辑）
    
    无论哪种模式，均使用网格法空间去自相关
    """
    rng = np.random.default_rng(seed)
    n_valid = valid_mask.sum()
    log(f"  有效像元总数: {n_valid}")

    # 各级分布
    class_counts = {}
    for c in range(1, n_classes + 1):
        count = int((agb_class[valid_mask] == c).sum())
        class_counts[c] = count
        log(f"  级别{c}: {count} 像元")

    if per_class_target > 0:
        log(f"  分层等量采样模式: 每级目标 {per_class_target}")
    else:
        log(f"  比例采样模式: {fraction*100:.0f}%")

    # 为每个类别独立采样
    all_rows, all_cols, all_labels = [], [], []

    for c in range(1, n_classes + 1):
        # 该类别的有效像元
        class_mask = valid_mask & (agb_class == c)
        class_rows, class_cols = np.where(class_mask)
        n_class = len(class_rows)

        if n_class == 0:
            log(f"  级别{c}: 无有效像元，跳过")
            continue

        # 确定该类别的目标采样数
        if per_class_target > 0:
            target = min(per_class_target, n_class)
        else:
            target = int(n_class * fraction)

        # 随机打乱后使用网格法空间去自相关
        order = rng.permutation(n_class)
        rows_s = class_rows[order]
        cols_s = class_cols[order]

        selected_r, selected_c = [], []
        grid_occupied = set()

        for i in range(n_class):
            r, c_pos = rows_s[i], cols_s[i]
            gk = (r // min_spacing, c_pos // min_spacing)
            conflict = False
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if (gk[0]+dr, gk[1]+dc) in grid_occupied:
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                selected_r.append(r)
                selected_c.append(c_pos)
                grid_occupied.add(gk)
                if len(selected_r) >= target:
                    break

        actual = len(selected_r)
        all_rows.extend(selected_r)
        all_cols.extend(selected_c)
        all_labels.extend([c] * actual)
        log(f"  级别{c}: 目标{target}, 实际采样{actual} (可用{n_class})")

    total = len(all_rows)
    log(f"  总采样: {total}")
    label_arr = np.array(all_labels)
    dist = dict(zip(*np.unique(label_arr, return_counts=True)))
    log(f"  采样后分布: {dist}")

    return np.array(all_rows), np.array(all_cols), np.array(all_labels)


def extract_samples(stack, agb_class, sample_rows, sample_cols, n_classes):
    """提取样本"""
    n_features = stack.shape[0]
    n_samples = len(sample_rows)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.full(n_samples, -1, dtype=np.int32)
    for i in range(n_samples):
        r, c = sample_rows[i], sample_cols[i]
        X[i, :] = stack[:, r, c]
        val = agb_class[r, c]
        if np.isfinite(val) and val > 0:
            y[i] = int(val)

    nan_ratio = np.isnan(X).sum(axis=1) / n_features
    valid = (nan_ratio < 0.3) & (y > 0) & (y <= n_classes)
    return X[valid], y[valid]


def main():
    cfg = parse_args()
    cfg.ensure_dirs()

    log("=" * 60)
    log(f"样本提取与RF分类建模 ({cfg.region_name})")
    log("=" * 60)
    log(f"质量筛选参数: slope<={cfg.slope_max}°, NDVI>={cfg.ndvi_min}")
    log(f"分层采样: per_class_target={cfg.per_class_target}")

    # 1. 加载
    log("[1/5] 加载特征栈和分级标签...")
    t0 = time.time()
    stack, feature_names, agb_class, valid_mask = load_feature_stack(cfg)
    log(f"  特征数: {len(feature_names)}, 加载耗时: {time.time()-t0:.1f}s")

    # 有效像元中各级分布
    n_total = valid_mask.sum()
    log(f"  质量筛选后有效像元: {n_total}")
    for c in range(1, cfg.n_classes + 1):
        n = (agb_class[valid_mask] == c).sum()
        pct = 100 * n / n_total if n_total > 0 else 0
        log(f"  级别{c}: {n} 像元 ({pct:.1f}%)")

    # 2. 分层采样
    log("[2/5] 分层等量空间采样...")
    t0 = time.time()
    sample_rows, sample_cols, sample_labels = stratified_spatial_sampling(
        valid_mask, agb_class, cfg.n_classes,
        min_spacing=cfg.min_spacing,
        per_class_target=cfg.per_class_target,
        fraction=cfg.sample_fraction,
        seed=cfg.random_seed
    )
    X, y = extract_samples(stack, agb_class, sample_rows, sample_cols, cfg.n_classes)
    del stack, agb_class, valid_mask
    log(f"  最终样本数: {len(y)}, 标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")
    log(f"  耗时: {time.time()-t0:.1f}s")

    # 3. 划分训练/验证
    log("[3/5] 划分训练/验证集 (70/30)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=cfg.random_seed, stratify=y
    )
    log(f"  训练: {len(X_train)}, 验证: {len(X_val)}")
    # 训练集分布
    train_dist = dict(zip(*np.unique(y_train, return_counts=True)))
    val_dist = dict(zip(*np.unique(y_val, return_counts=True)))
    log(f"  训练集分布: {train_dist}")
    log(f"  验证集分布: {val_dist}")

    # 4. 训练RF分类器
    log("[4/5] 训练RF分类器...")
    t0 = time.time()

    # 构建参数（处理max_depth=None的情况）
    rf_params = {}
    for k, v in cfg.rf_params.items():
        rf_params[k] = None if v == "null" or v is None else v

    rf_final = RandomForestClassifier(
        **rf_params,
        random_state=cfg.random_seed,
        n_jobs=-1,
        class_weight="balanced"
    )

    # CV
    cv_folds = cfg.cv_folds
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=cfg.random_seed)
    cv_scores = cross_val_score(rf_final, X_train, y_train, cv=kfold, scoring="accuracy", n_jobs=-1)
    log(f"  {cv_folds}折CV OA: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    rf_final.fit(X_train, y_train)
    log(f"  训练耗时: {time.time()-t0:.1f}s")

    # 5. 独立验证
    log("[5/5] 独立验证...")
    t0 = time.time()

    y_val_pred = rf_final.predict(X_val)
    oa = accuracy_score(y_val, y_val_pred)
    kappa = cohen_kappa_score(y_val, y_val_pred)

    log(f"  === 独立验证指标 ===")
    log(f"  OA: {oa:.4f}")
    log(f"  Kappa: {kappa:.4f}")

    class_names = [f"Level{i+1}" for i in range(cfg.n_classes)]
    report = classification_report(y_val, y_val_pred, target_names=class_names, digits=4)
    log(f"  分类报告:\n{report}")

    cm = confusion_matrix(y_val, y_val_pred)
    log(f"  混淆矩阵:\n{cm}")

    # 特征重要性
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_final.feature_importances_
    }).sort_values("importance", ascending=False)

    log(f"  特征重要性 Top 5:")
    for _, row in fi.head(5).iterrows():
        log(f"    {row['feature']}: {row['importance']:.4f}")

    log(f"  验证耗时: {time.time()-t0:.1f}s")

    # 保存
    log("保存结果...")
    joblib.dump(rf_final, cfg.rf_model)

    cm_norm_pa = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm_ua = cm.astype(float) / cm.sum(axis=0, keepdims=True)
    pa_list = [float(cm_norm_pa[i, i]) for i in range(cfg.n_classes)]
    ua_list = [float(cm_norm_ua[i, i]) for i in range(cfg.n_classes)]

    metrics = {
        "model_type": "RandomForestClassifier",
        "region": cfg.region_id,
        "quality_filter": {
            "slope_max": cfg.slope_max,
            "ndvi_min": cfg.ndvi_min,
            "per_class_target": cfg.per_class_target,
        },
        "best_params": {k: str(v) for k, v in rf_params.items()},
        "cv_folds": cv_folds,
        "selected_features": cfg.selected_features if cfg.selected_features else "all",
        "cv_oa_mean": float(cv_scores.mean()),
        "cv_oa_std": float(cv_scores.std()),
        "cv_oa_folds": [float(s) for s in cv_scores],
        "val_oa": float(oa),
        "val_kappa": float(kappa),
        "pa_per_class": {f"Level{i+1}": pa_list[i] for i in range(cfg.n_classes)},
        "ua_per_class": {f"Level{i+1}": ua_list[i] for i in range(cfg.n_classes)},
        "confusion_matrix": cm.tolist(),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": len(feature_names),
        "n_classes": cfg.n_classes,
    }
    with open(cfg.model_metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    fi.to_csv(cfg.feature_importance_csv, index=False)

    np.savez(cfg.validation_data,
             y_val=y_val, y_val_pred=y_val_pred,
             y_train=y_train, y_train_pred=rf_final.predict(X_train),
             confusion_matrix=cm)

    log("RF分类建模完成!")


if __name__ == "__main__":
    main()
