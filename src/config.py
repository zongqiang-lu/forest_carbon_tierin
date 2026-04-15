"""
统一配置管理模块
================
功能：读取YAML配置文件，提供研究区参数和路径的统一访问接口
支持多年份：region_id格式为 {region}_{year}，如 ninger_2019

用法：
    from config import load_config
    cfg = load_config("ninger_2019")
    print(cfg.boundary)       # data/ninger/raw/boundary/宁洱县_32647.geojson
    print(cfg.intermediate_dir)  # data/ninger/2019/intermediate
"""

import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import yaml


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass
class RegionConfig:
    """研究区配置（含年份）"""
    region_id: str       # e.g. "ninger"
    region_name: str     # e.g. "宁洱县"
    year: int            # e.g. 2023
    crs: str
    resolution: float

    # 原始数据路径 (绝对路径)
    boundary: Path
    dem: Path
    agb: Optional[Path]  # 2023-2025年无AGB标签时为None
    landsat_dir: Path
    clcd: Path           # 对应年份的CLCD数据

    # AGB参数
    agb_nodata: float = 0
    agb_max_valid: float = 500
    n_classes: int = 5

    # RF参数
    rf_params: Dict[str, Any] = field(default_factory=dict)

    # 采样参数
    sampling: Dict[str, Any] = field(default_factory=dict)

    # 样本质量筛选参数
    quality_filter: Dict[str, Any] = field(default_factory=dict)

    # 特征筛选（可选，为空则使用全部特征）
    selected_features: list = field(default_factory=list)

    # CV折数
    cv_folds: int = 5

    # 是否强制重算（跳过中间产物存在性检查）
    force: bool = False

    @property
    def has_agb(self) -> bool:
        """是否有AGB标签数据"""
        return self.agb is not None and self.agb.exists()

    @property
    def clcd_cache_dir(self) -> Path:
        """CLCD缓存目录"""
        d = PROJECT_ROOT / "data" / "shared" / "clcd_cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def clcd_forest_mask_cache(self) -> Path:
        """CLCD森林掩膜缓存路径"""
        return self.clcd_cache_dir / f"{self.region_id}_{self.year}_forest_mask.tif"

    def find_existing_terrain_stack(self) -> Path | None:
        """查找同研究区其他年份已存在的terrain_stack（地形不随年份变化）"""
        for existing_year in [2019, 2023, 2024, 2025]:
            if existing_year == self.year:
                continue
            candidate = (PROJECT_ROOT / "data" / self.region_id
                         / str(existing_year) / "intermediate" / "terrain_stack.tif")
            if candidate.exists():
                return candidate
        return None

    @property
    def data_dir(self) -> Path:
        """研究区年份数据根目录: data/{region}/{year}"""
        return PROJECT_ROOT / "data" / self.region_id / str(self.year)

    @property
    def intermediate_dir(self) -> Path:
        """中间产物目录"""
        return self.data_dir / "intermediate"

    @property
    def aligned_dir(self) -> Path:
        """对齐后数据目录"""
        return self.data_dir / "aligned"

    @property
    def output_dir(self) -> Path:
        """输出根目录: output/{region}/{year}"""
        return PROJECT_ROOT / "output" / self.region_id / str(self.year)

    @property
    def metrics_dir(self) -> Path:
        """模型指标目录"""
        return self.output_dir / "metrics"

    @property
    def figures_dir(self) -> Path:
        """图表目录"""
        return self.output_dir / "figures"

    @property
    def classification_dir(self) -> Path:
        """分类结果目录"""
        return self.output_dir / "classification"

    # --- 中间产物路径快捷方式 ---

    @property
    def landsat_sr_stack(self) -> Path:
        return self.intermediate_dir / "landsat_sr_stack.tif"

    @property
    def quality_mask(self) -> Path:
        return self.intermediate_dir / "quality_mask.tif"

    @property
    def terrain_stack(self) -> Path:
        return self.intermediate_dir / "terrain_stack.tif"

    @property
    def indices_stack(self) -> Path:
        return self.intermediate_dir / "indices_stack.tif"

    # --- 对齐后数据路径 ---

    @property
    def feature_stack(self) -> Path:
        return self.aligned_dir / "feature_stack.tif"

    @property
    def feature_names_json(self) -> Path:
        return self.aligned_dir / "feature_names.json"

    @property
    def valid_mask(self) -> Path:
        return self.aligned_dir / "valid_mask.tif"

    @property
    def forest_mask(self) -> Path:
        return self.aligned_dir / "forest_mask.tif"

    @property
    def agb_continuous(self) -> Path:
        return self.aligned_dir / "agb_continuous.tif"

    @property
    def agb_class(self) -> Path:
        return self.aligned_dir / f"agb_class_{self.n_classes}level.tif"

    @property
    def jenks_breaks_json(self) -> Path:
        return self.aligned_dir / "jenks_breaks.json"

    # --- 输出路径 ---

    @property
    def rf_model(self) -> Path:
        return self.metrics_dir / "rf_model.pkl"

    @property
    def model_metrics_json(self) -> Path:
        return self.metrics_dir / "model_metrics.json"

    @property
    def feature_importance_csv(self) -> Path:
        return self.metrics_dir / "feature_importance.csv"

    @property
    def validation_data(self) -> Path:
        return self.metrics_dir / "validation_data.npz"

    @property
    def classification_tif(self) -> Path:
        return self.classification_dir / "carbon_density_class.tif"

    @property
    def classification_breaks_json(self) -> Path:
        return self.classification_dir / "classification_breaks.json"

    # --- 采样参数快捷方式 ---

    @property
    def min_spacing(self) -> int:
        return self.sampling.get("min_spacing", 3)

    @property
    def sample_fraction(self) -> float:
        return self.sampling.get("fraction", 0.15)

    @property
    def random_seed(self) -> int:
        return self.sampling.get("seed", 42)

    # --- 质量筛选参数快捷方式 ---

    @property
    def slope_max(self) -> float:
        """坡度阈值(度)，超过则剔除"""
        return self.quality_filter.get("slope_max", 35)

    @property
    def ndvi_min(self) -> float:
        """NDVI最小阈值，低于则剔除"""
        return self.quality_filter.get("ndvi_min", 0.3)

    @property
    def per_class_target(self) -> int:
        """分层采样每级目标样本数，0=不限制"""
        return self.quality_filter.get("per_class_target", 0)

    @property
    def quality_mask_filtered(self) -> Path:
        """质量筛选后的综合掩膜"""
        return self.aligned_dir / "quality_filtered_mask.tif"

    # --- 跨年份模型引用 ---

    @property
    def model_region_id(self) -> str:
        """用于预测时引用的模型来源研究区（默认与当前相同）"""
        return self.region_id

    @property
    def model_year(self) -> int:
        """用于预测时引用的模型年份（默认2019）"""
        return self._model_year if hasattr(self, '_model_year') else 2019

    @model_year.setter
    def model_year(self, value: int):
        self._model_year = value

    def get_model_path(self) -> Path:
        """获取用于预测的模型路径（支持跨年份/跨研究区）"""
        return (PROJECT_ROOT / "output" / self.model_region_id
                / str(self.model_year) / "metrics" / "rf_model.pkl")

    def get_jenks_breaks_path(self) -> Path:
        """获取用于预测的Jenks断点路径"""
        return (PROJECT_ROOT / "data" / self.model_region_id
                / str(self.model_year) / "aligned" / "jenks_breaks.json")

    def ensure_dirs(self):
        """确保所有输出目录存在"""
        for d in [
            self.intermediate_dir, self.aligned_dir,
            self.metrics_dir, self.figures_dir, self.classification_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


def load_config(region_year: str, model_year: int = None) -> RegionConfig:
    """
    读取研究区YAML配置文件，返回RegionConfig对象

    Args:
        region_year: 研究区+年份ID，格式为 {region}_{year}，如 ninger_2019
                     也兼容纯region_id（默认year=2019）
        model_year: 预测时使用的模型年份，默认None表示与当前year相同

    Returns:
        RegionConfig实例
    """
    # 解析region_id和year
    parts = region_year.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        region_id = parts[0]
        year = int(parts[1])
    else:
        region_id = region_year
        year = 2019  # 兼容旧格式

    yaml_path = CONFIG_DIR / f"{region_id}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"配置文件不存在: {yaml_path}\n"
            f"可用配置: {[p.stem for p in CONFIG_DIR.glob('*.yaml')]}"
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # 解析AGB路径（仅2019年使用PKU-AGB标签，其他年份agb=None）
    agb_rel = raw.get("agb")
    if year == 2019 and agb_rel:
        agb_rel_resolved = agb_rel.replace("{year}", str(year))
        agb_path_resolved = PROJECT_ROOT / agb_rel_resolved
    else:
        agb_path_resolved = None

    # CLCD路径：从shared目录按年份选择
    clcd_rel = raw.get("clcd")
    if clcd_rel:
        # 如果yaml中指定了clcd模板，用year替换{year}
        clcd_path = PROJECT_ROOT / clcd_rel.format(year=year)
    else:
        # 默认路径
        clcd_path = PROJECT_ROOT / f"data/shared/clcd/CLCD_v01_{year}_albert.tif"

    # 解析路径（替换{year}占位符）
    def resolve_path(rel_path):
        if rel_path is None:
            return None
        return PROJECT_ROOT / rel_path.format(year=year, region=region_id)

    # 如果boundary/landsat_dir中有{year}，自动替换
    boundary_rel = raw.get("boundary", "").replace("{year}", str(year))
    landsat_dir_rel = raw.get("landsat_dir", "").replace("{year}", str(year))

    cfg = RegionConfig(
        region_id=raw["region_id"],
        region_name=raw["region_name"],
        year=year,
        crs=raw["crs"],
        resolution=raw["resolution"],
        boundary=PROJECT_ROOT / boundary_rel,
        dem=PROJECT_ROOT / raw["dem"],
        agb=agb_path_resolved,
        landsat_dir=PROJECT_ROOT / landsat_dir_rel,
        clcd=clcd_path,
        agb_nodata=raw.get("agb_nodata", 0),
        agb_max_valid=raw.get("agb_max_valid", 500),
        n_classes=raw.get("n_classes", 5),
        rf_params=raw.get("rf_params", {}),
        sampling=raw.get("sampling", {}),
        quality_filter=raw.get("quality_filter", {}),
        selected_features=raw.get("selected_features", []),
        cv_folds=raw.get("cv_folds", 5),
    )

    # 设置模型年份
    if model_year is not None:
        cfg.model_year = model_year
    elif year != 2019:
        # 非2019年默认使用2019模型
        cfg.model_year = 2019

    return cfg


def parse_args() -> RegionConfig:
    """
    解析命令行参数并加载配置

    用法: python script.py --region ninger --year 2019
          python script.py --region ninger --year 2025 --model_year 2019
    """
    parser = argparse.ArgumentParser(description="树智碳汇")
    parser.add_argument(
        "--region", type=str, default="ninger",
        help="研究区ID (对应config/{region}.yaml)"
    )
    parser.add_argument(
        "--year", type=int, default=2019,
        help="数据年份 (默认2019)"
    )
    parser.add_argument(
        "--model_year", type=int, default=None,
        help="预测时使用的模型年份 (默认: 2019年数据使用自身模型, 其他年份使用2019模型)"
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="强制重算，跳过中间产物存在性检查"
    )
    args = parser.parse_args()
    region_year = f"{args.region}_{args.year}"
    cfg = load_config(region_year, model_year=args.model_year)
    cfg.force = args.force
    return cfg
