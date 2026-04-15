"""
一键运行脚本 - 树智碳汇全流程
===================================
按顺序执行所有处理步骤

用法:
    python src/run_all.py --region ninger --year 2019     # 完整流程（有AGB标签）
    python src/run_all.py --region ninger --year 2023     # 预测模式（使用2019模型）
    python src/run_all.py --region ninger --year 2024 --model_year 2019
"""

import subprocess
import sys
from pathlib import Path

# 添加src目录到路径以导入config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

# 完整流程（2019年，有AGB标签）
FULL_PIPELINE = [
    "01_landsat_preprocess.py",   # Landsat L2SP预处理 (多景中值合成)
    "02_dem_preprocess.py",       # DEM预处理 (8波段terrain_stack)
    "03_vegetation_indices.py",   # 植被指数计算 (13波段indices_stack)
    "04_feature_stack.py",        # 特征栈构建 (28波段feature_stack)
    "05_label_preparation.py",    # PKU-AGB标签制备 (含边界模糊过滤)
    "06_sample_model.py",         # 样本提取与RF建模
    "06b_hierarchical_classifier.py",  # 分层二分类器建模
    "07_prediction_viz.py",       # AGB反演与分级制图
]

# 预测流程（非基准年，无AGB标签，使用已有模型）
PREDICT_PIPELINE = [
    "01_landsat_preprocess.py",   # Landsat L2SP预处理
    "02_dem_preprocess.py",       # DEM预处理 (可复用基准年的)
    "03_vegetation_indices.py",   # 植被指数计算
    "04_feature_stack.py",        # 特征栈构建 (CLCD森林掩膜)
    "07_prediction_viz.py",       # 用已有模型预测
]


def run_script(script_name, region_id, year, model_year=None, force=False):
    """运行单个脚本"""
    script_path = SRC_DIR / script_name
    if not script_path.exists():
        print(f"  跳过不存在的脚本: {script_name}")
        return True  # 不视为失败

    cmd = [sys.executable, str(script_path), "--region", region_id, "--year", str(year)]
    if model_year is not None:
        cmd.extend(["--model_year", str(model_year)])
    if force:
        cmd.append("--force")

    print(f"\n{'=' * 70}")
    print(f"运行: {script_name} --region {region_id} --year {year}"
          + (f" --model_year {model_year}" if model_year else "")
          + (" --force" if force else ""))
    print(f"{'=' * 70}")

    result = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"X {script_name} 执行失败 (返回码: {result.returncode})")
        return False
    else:
        print(f"  {script_name} 执行成功")
        return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="树智碳汇全流程运行")
    parser.add_argument("--region", type=str, default="ninger",
                        help="研究区ID (对应config/{region}.yaml)")
    parser.add_argument("--year", type=int, default=2019,
                        help="数据年份 (默认2019)")
    parser.add_argument("--model_year", type=int, default=None,
                        help="预测时使用的模型年份 (默认: 基准年用自身模型, 其他用2019)")
    parser.add_argument("--force", action="store_true", default=False,
                        help="强制重算，跳过中间产物存在性检查")
    args = parser.parse_args()

    # 加载配置验证
    region_year = f"{args.region}_{args.year}"
    cfg = load_config(region_year, model_year=args.model_year)

    # 确定流程
    if cfg.has_agb:
        scripts = FULL_PIPELINE
        mode = "完整流程 (有AGB标签)"
    else:
        scripts = PREDICT_PIPELINE
        mode = f"预测模式 (使用{cfg.model_year}年模型)"

    print("=" * 70)
    print(f"树智碳汇 - 全流程运行")
    print(f"研究区: {cfg.region_name} ({cfg.region_id}), 年份: {cfg.year}")
    print(f"坐标系: {cfg.crs}, 分辨率: {cfg.resolution}m")
    print(f"模式: {mode}")
    print(f"步骤: {len(scripts)}步")
    print("=" * 70)

    success_count = 0
    for script in scripts:
        if run_script(script, args.region, args.year, args.model_year, args.force):
            success_count += 1
        else:
            print(f"\n  {script} 失败，是否继续？(后续步骤可能依赖该步骤输出)")

    print(f"\n{'=' * 70}")
    print(f"流程完成: {success_count}/{len(scripts)} 步骤成功")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
