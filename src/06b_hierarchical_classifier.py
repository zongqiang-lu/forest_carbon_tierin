"""
分层二分类器
============
功能：将三分类问题拆解为两个二分类任务，降低类别混淆

策略：
- 第一层：Level1 vs Level2+3 二分类器
- 第二层：仅对非Level1样本做 Level2 vs Level3 二分类

优势：减少Level2/Level3边界混淆，提升过渡带分类精度
"""

import sys
import time
import numpy as np
import rasterio
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
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
    """加载特征栈和AGB分级标签"""
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
        selected_indices = [
            all_features.index(f) for f in cfg.selected_features if f in all_features
        ]
        stack = stack[selected_indices]
        all_features = [all_features[i] for i in selected_indices]
        log(
            f"  特征筛选: {meta['n_features']} -> {len(all_features)} ({len(all_features)}/{meta['n_features']})"
        )

    return stack, all_features, agb_class, valid_mask


def stratified_spatial_sampling(
    valid_mask,
    agb_class,
    n_classes,
    min_spacing=5,
    per_class_target=0,
    fraction=0.15,
    seed=42,
):
    """分层等量空间采样"""
    rng = np.random.default_rng(seed)
    all_rows, all_cols, all_labels = [], [], []

    for c in range(1, n_classes + 1):
        class_mask = valid_mask & (agb_class == c)
        class_rows, class_cols = np.where(class_mask)
        n_class = len(class_rows)

        if n_class == 0:
            continue

        if per_class_target > 0:
            target = min(per_class_target, n_class)
        else:
            target = int(n_class * fraction)

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
                    if (gk[0] + dr, gk[1] + dc) in grid_occupied:
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

        all_rows.extend(selected_r)
        all_cols.extend(selected_c)
        all_labels.extend([c] * len(selected_r))

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


class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    """
    分层二分类器
    策略：Level1 vs Level2+3 → Level2 vs Level3
    """

    def __init__(self, rf_params=None, seed=42):
        self.seed = seed
        self.rf_params = rf_params or {}
        self.clf_1_vs_23 = RandomForestClassifier(
            **self.rf_params, random_state=self.seed, n_jobs=-1, class_weight="balanced"
        )
        self.clf_2_vs_3 = RandomForestClassifier(
            **self.rf_params, random_state=self.seed, n_jobs=-1, class_weight="balanced"
        )
        # 兼容sklearn接口
        self.n_features_in_ = None
        self.classes_ = np.array([1, 2, 3])

    def get_params(self, deep=True):
        return {"rf_params": self.rf_params, "seed": self.seed}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        """分层训练"""
        log("  第一层训练: Level1 vs Level2+3...")
        y_binary_1 = (y == 1).astype(int)
        self.clf_1_vs_23.fit(X, y_binary_1)

        log("  第二层训练: Level2 vs Level3...")
        mask_23 = y >= 2
        if mask_23.sum() > 100:
            X_23 = X[mask_23]
            y_23 = y[mask_23]
            y_binary_23 = (y_23 == 3).astype(int)
            self.clf_2_vs_3.fit(X_23, y_binary_23)

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array(sorted(set(y.astype(int))))
        return self

    def predict(self, X):
        """分层预测"""
        pred_1 = self.clf_1_vs_23.predict(X)
        pred_23 = self.clf_2_vs_3.predict(X)
        result = np.where(pred_1 == 1, 1, np.where(pred_23 == 1, 3, 2))
        return result

    def predict_proba(self, X):
        """分层预测概率"""
        prob_1 = self.clf_1_vs_23.predict_proba(X)[:, 1]
        prob_23 = np.zeros(len(X))
        if hasattr(self.clf_2_vs_3, "classes_"):
            if len(self.clf_2_vs_3.classes_) == 2:
                prob_23 = self.clf_2_vs_3.predict_proba(X)[:, 0]

        final_proba = np.zeros((len(X), 3))
        final_proba[:, 0] = prob_1
        final_proba[:, 1] = (1 - prob_1) * prob_23
        final_proba[:, 2] = (1 - prob_1) * (1 - prob_23)
        return final_proba


def main():
    cfg = parse_args()
    cfg.ensure_dirs()

    log("=" * 60)
    log(f"分层二分类训练 ({cfg.region_name})")
    log("=" * 60)
    log(f"质量筛选: slope<={cfg.slope_max}°, NDVI>={cfg.ndvi_min}")
    log(f"分层采样: per_class_target={cfg.per_class_target}")

    log("[1/5] 加载特征栈和分级标签...")
    t0 = time.time()
    stack, feature_names, agb_class, valid_mask = load_feature_stack(cfg)
    log(f"  特征数: {len(feature_names)}, 加载耗时: {time.time() - t0:.1f}s")

    n_total = valid_mask.sum()
    log(f"  质量筛选后有效像元: {n_total}")
    for c in range(1, cfg.n_classes + 1):
        n = (agb_class[valid_mask] == c).sum()
        pct = 100 * n / n_total if n_total > 0 else 0
        log(f"  级别{c}: {n} 像元 ({pct:.1f}%)")

    log("[2/5] 分层等量空间采样...")
    t0 = time.time()
    sample_rows, sample_cols, sample_labels = stratified_spatial_sampling(
        valid_mask,
        agb_class,
        cfg.n_classes,
        min_spacing=cfg.min_spacing,
        per_class_target=cfg.per_class_target,
        fraction=cfg.sample_fraction,
        seed=cfg.random_seed,
    )
    X, y = extract_samples(stack, agb_class, sample_rows, sample_cols, cfg.n_classes)
    del stack, agb_class, valid_mask
    log(
        f"  最终样本数: {len(y)}, 标签分布: {dict(zip(*np.unique(y, return_counts=True)))}"
    )
    log(f"  耗时: {time.time() - t0:.1f}s")

    log("[3/5] 划分训练/验证集 (70/30)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=cfg.random_seed, stratify=y
    )
    log(f"  训练: {len(X_train)}, 验证: {len(X_val)}")

    log("[4/5] 训练分层二分类器...")
    t0 = time.time()

    rf_params = {}
    for k, v in cfg.rf_params.items():
        rf_params[k] = None if v == "null" or v is None else v

    clf = HierarchicalClassifier(rf_params, seed=cfg.random_seed)
    clf.fit(X_train, y_train)
    log(f"  训练耗时: {time.time() - t0:.1f}s")

    # 交叉验证
    cv_folds = cfg.cv_folds
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=cfg.random_seed)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=kfold, scoring="accuracy", n_jobs=-1)
    log(f"  {cv_folds}折CV OA: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    log("[5/5] 独立验证...")
    t0 = time.time()

    y_val_pred = clf.predict(X_val)
    oa = accuracy_score(y_val, y_val_pred)
    kappa = cohen_kappa_score(y_val, y_val_pred)

    log(f"  === 分层二分类验证指标 ===")
    log(f"  OA: {oa:.4f}")
    log(f"  Kappa: {kappa:.4f}")

    class_names = [f"Level{i + 1}" for i in range(cfg.n_classes)]
    report = classification_report(
        y_val, y_val_pred, target_names=class_names, digits=4
    )
    log(f"  分类报告:\n{report}")

    cm = confusion_matrix(y_val, y_val_pred)
    log(f"  混淆矩阵:\n{cm}")

    fi_1 = pd.DataFrame(
        {"feature": feature_names, "importance": clf.clf_1_vs_23.feature_importances_}
    ).sort_values("importance", ascending=False)

    log(f"  第一层(L1 vs L23)特征重要性 Top 5:")
    for _, row in fi_1.head(5).iterrows():
        log(f"    {row['feature']}: {row['importance']:.4f}")

    log(f"  验证耗时: {time.time() - t0:.1f}s")

    log("保存结果...")
    joblib.dump(clf, cfg.rf_model)

    cm_norm_pa = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm_ua = cm.astype(float) / cm.sum(axis=0, keepdims=True)
    pa_list = [float(cm_norm_pa[i, i]) for i in range(cfg.n_classes)]
    ua_list = [float(cm_norm_ua[i, i]) for i in range(cfg.n_classes)]

    metrics = {
        "model_type": "HierarchicalClassifier (分层二分类)",
        "region": cfg.region_id,
        "quality_filter": {
            "slope_max": cfg.slope_max,
            "ndvi_min": cfg.ndvi_min,
            "per_class_target": cfg.per_class_target,
        },
        "best_params": {k: str(v) for k, v in rf_params.items()},
        "cv_folds": cv_folds,
        "cv_oa_mean": float(cv_scores.mean()),
        "cv_oa_std": float(cv_scores.std()),
        "cv_oa_folds": [float(s) for s in cv_scores],
        "selected_features": list(feature_names) if not cfg.selected_features else cfg.selected_features,
        "class_names": ["低碳密度", "中碳密度", "高碳密度"],
        "val_oa": float(oa),
        "val_kappa": float(kappa),
        "pa_per_class": {f"Level{i + 1}": pa_list[i] for i in range(cfg.n_classes)},
        "ua_per_class": {f"Level{i + 1}": ua_list[i] for i in range(cfg.n_classes)},
        "confusion_matrix": cm.tolist(),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": len(feature_names),
        "n_classes": cfg.n_classes,
    }
    with open(cfg.model_metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    fi_1.to_csv(cfg.feature_importance_csv, index=False)

    log("分层二分类建模完成!")


if __name__ == "__main__":
    main()
