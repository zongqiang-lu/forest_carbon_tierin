# 树智碳汇 (TreeWit Carbon)

基于 Landsat 遥感影像和分层二分类器的森林碳汇密度分级与监测平台，支持多研究区、多年份的碳汇密度预测与可视化。

## 功能特性

- **8步处理流水线**: Landsat预处理 → DEM预处理(8因子) → 植被指数(13种) → 特征堆叠(28维) → 标签制备(含边界过滤) → RF建模 → 分层分类器建模 → 预测可视化
- **28维特征**: 7个Landsat光谱波段 + 13种植被指数(含缨帽变换) + 8个地形因子(含曲率/坡位/粗糙度)
- **分层二分类器**: HierarchicalClassifier — L1 vs L23, L2 vs L3，比直接三分类更合理
- **边界模糊过滤**: Jenks断点±8 t/ha范围样本剔除，减少边界等级混淆
- **Jenks Natural Breaks分类**: 自动将连续AGB分为低碳/中碳/高碳密度等级
- **PKU-AGB数据源**: 使用北京大学发布的2019年森林AGB数据
- **CLCD森林掩膜**: 基于中国土地覆盖数据自动提取森林区域
- **跨年份预测**: 使用2019年训练模型对2023/2024/2025年数据进行预测
- **Web可视化平台**: FastAPI + Vue3 + Leaflet 在线展示
- **本地GUI工具**: CustomTkinter 浅色主题桌面应用

## 项目结构

```
forest_carbon_tierin/
├── config/                    # 研究区配置 (YAML)
│   ├── credentials/           # API凭据 (gitignore)
│   ├── ninger.yaml            # 宁洱县配置
│   └── shuangbai.yaml         # 双柏县配置
├── src/                       # 核心处理流水线
│   ├── config.py              # 统一配置管理
│   ├── run_all.py             # 一键运行脚本
│   ├── 01_landsat_preprocess.py
│   ├── 02_dem_preprocess.py   # 8波段地形因子
│   ├── 03_vegetation_indices.py  # 13波段植被指数
│   ├── 04_feature_stack.py    # 28维特征栈 (WarpedVRT)
│   ├── 05_label_preparation.py   # 含边界模糊过滤
│   ├── 06_sample_model.py     # RF分类器
│   ├── 06b_hierarchical_classifier.py  # 分层二分类器
│   └── 07_prediction_viz.py   # 支持HC/RF双模型
├── gui/                       # CustomTkinter GUI
│   ├── app.py                 # 主入口
│   ├── styles.py              # 共享主题
│   ├── prediction_tab.py      # 碳汇预测
│   ├── landsat_tab.py         # Landsat下载
│   ├── geojson_tab.py         # GeoJSON转换
│   ├── gedi_tab.py            # GEDI下载
│   └── web_service_tab.py     # Web服务管理
├── web/                       # Web平台
│   ├── backend/               # FastAPI后端
│   ├── frontend/              # Vue3前端
│   └── scripts/               # 数据预处理
├── data/                      # 数据目录 (gitignore)
└── output/                    # 输出目录 (gitignore)
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourname/forest_carbon_tierin.git
cd forest_carbon_tierin

# 安装依赖
pip install -r requirements.txt
```

## 使用

### 启动本地GUI

```bash
python -m gui.app
```

或双击 `start.bat`

### 命令行运行流水线

```bash
# 完整流水线 (2019年, 含训练)
python -m src.run_all --region ninger --year 2019

# 预测流水线 (2023/2024/2025年, 使用2019模型)
python -m src.run_all --region ninger --year 2023
```

### 启动Web平台

```bash
python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## 技术栈

- **核心**: Python, rasterio, scikit-learn, geopandas
- **GUI**: CustomTkinter (浅色主题)
- **Web**: FastAPI + Vue3 + Leaflet + ECharts
- **分类器**: HierarchicalClassifier (分层二分类) / RandomForest
- **分级方法**: Jenks Natural Breaks
- **AGB数据源**: 北京大学遥感与GIS研究所 (2019)

## License

MIT License
