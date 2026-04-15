# 树智碳汇 (TreeWit Carbon) — 项目总结

## 一、项目概述

基于 Landsat 遥感影像和分层二分类器的**森林碳汇密度分级与时空动态监测平台**。针对云南省宁洱哈尼族彝族自治县和双柏县，以2019年为基准训练年，实现多年份碳汇密度预测、分级制图与可视化分析。面向大学生计算机设计大赛 AI 应用赛道。

## 二、技术架构

项目采用三大模块架构：核心处理流水线 (`src/`)、本地 GUI (`gui/`)、Web 平台 (`web/`)。

### 2.1 核心处理流水线

| 步骤 | 脚本 | 输入 | 输出 | 功能 |
|------|------|------|------|------|
| 1 | `01_landsat_preprocess.py` | Landsat L2SP 原始数据 | `landsat_sr_stack.tif` (7波段) + `quality_mask.tif` | 多景自动发现、QA像素解析、中值合成无云影像 |
| 2 | `02_dem_preprocess.py` | SRTM DEM | `terrain_stack.tif` (8波段: 海拔/坡度/坡向/TWI/剖面曲率/平面曲率/坡位/粗糙度) | DEM重投影重采样、简化TWI近似、曲率/坡位/粗糙度计算、支持跨年复用 |
| 3 | `03_vegetation_indices.py` | landsat_sr_stack | `indices_stack.tif` (13波段: NDVI/EVI/NDMI/SAVI/NDWI/MSAVI/NBR/SR_B5B4/SR_B6B5/SR_B7B5/TCB/TCG/TCW) | 13种植被指数计算（含缨帽变换） |
| 4 | `04_feature_stack.py` | 3个stack + CLCD + 边界 | `feature_stack.tif` (28波段) + 各种掩膜 | WarpedVRT批量对齐、CLCD森林掩膜(value=2)、AGB范围过滤、低质量样本过滤(全局相关性<0.1)、28维特征合并 |
| 5 | `05_label_preparation.py` | AGB连续值 + 森林掩膜 | `agb_class_3level.tif` + `jenks_breaks.json` | Jenks自然断点分级(3级)、边界模糊样本过滤(±8 t/ha) |
| 6 | `06_sample_model.py` | 特征栈 + AGB分级标签 | `rf_model.pkl` + 模型指标 | 分层等量空间采样、RF分类器(5-fold CV + 30%独立验证)、允许30% NaN特征 |
| 6b | `06b_hierarchical_classifier.py` | 特征栈 + AGB分级标签 | `hc_model.pkl` + 模型指标 | 分层二分类器(L1 vs L23, L2 vs L3)、联合概率估计 |
| 7 | `07_prediction_viz.py` | 特征栈 + 模型 | `carbon_density_class.tif` + 可视化图表 | 全区域分类预测(批量)、支持RF和HierarchicalClassifier、分级专题图 |

**双模式运行**：
- **完整流程** (2019年，有AGB标签)：8步全执行，训练模型
- **预测流程** (2023/2024/2025年，无AGB)：仅执行1-4+7步，复用2019模型

`run_all.py` 一键运行，自动根据 `has_agb` 属性选择流程。

### 2.2 统一配置管理

`RegionConfig` 数据类 (`src/config.py`) 管理所有参数，从 YAML 加载：
- 路径管理：`{year}`占位符自动替换、`find_existing_terrain_stack()`跨年复用地形数据
- 模型引用：`model_year`属性支持跨年份/跨研究区模型引用（默认2019）
- 采样参数：`min_spacing`(空间去自相关网格)、`per_class_target`(分层等量)、`slope_max`/`ndvi_min`(质量筛选)
- 特征筛选：`selected_features`为空时使用全部28维特征

### 2.3 本地 GUI

基于 **CustomTkinter** 的浅色主题桌面应用，5个Tab页：

| Tab | 文件 | 功能 |
|-----|------|------|
| 碳汇预测 | `prediction_tab.py` | 研究区/年份选择(2019-2025)、模型加载(HC/RF)、一键预测(异步)、matplotlib可视化、矢量边界掩膜、统计信息、TIF/PNG导出 |
| Landsat下载 | `landsat_tab.py` | USGS M2M API认证、产品ID查询、SR波段筛选、多线程并发下载、凭据持久化 |
| GeoJSON转换 | `geojson_tab.py` | EPSG坐标系转换(如CGCS2000→UTM)、基于geopandas |
| GEDI下载 | `gedi_tab.py` | EarthData认证、earthaccess搜索、Harmony空间子集化下载、H5数据处理(8波束11变量)、质量过滤+AOI裁剪→GPKG |
| Web服务 | `web_service_tab.py` | FastAPI后端启停、数据预处理触发、浏览器打开、服务日志监控 |

### 2.4 Web 平台

#### 后端 (FastAPI)

- **地图API**：分类图/变化图/置信度图的PNG和bounds、边界GeoJSON
- **统计API**：概览统计、变化检测统计、区域年度统计、置信度统计
- **模型API**：评估指标(OA/Kappa/PA/UA)、特征重要性、分类断点
- **点查询API** (`POST /api/query/point`)：根据经纬度查询碳汇等级、特征值、多年对比、置信度
- **AI聊天** (`/api/chat/stream`)：SSE流式端点，基于DeepSeek大模型，注入项目数据上下文

#### 前端 (Vue3 + Element Plus + Leaflet + ECharts)

- **首页**：项目概览、统计数据卡片、技术路线流程图、研究区概况
- **时空监测页**：Leaflet地图、天地图底图(卫星/矢量/暗色)、碳汇分级/变化检测/置信度图层切换、分屏对比(同步联动+拖拽)、点查询面板
- **统计分析页**：饼图/柱状图/趋势图/特征重要性图/双区对比图
- **不确定性页**：模型评估指标、各等级PA/UA、置信度分布、误差来源分析
- **模型详情页**：参数配置表、特征重要性图、混淆矩阵热力图、CV折线图
- **AI助手**：浮动按钮+聊天面板、SSE流式响应、Markdown渲染

#### 数据预处理

`generate_web_data.py` 5步预处理：分类TIF→WGS84重投影+彩色PNG → 统计信息预计算 → 变化检测图 → RF置信度图 → 边界GeoJSON

## 三、数据管线

```
Landsat L2SP → 多景中值合成(7波段SR)
SRTM DEM → 重投影重采样 → 8波段地形因子(含曲率/坡位/粗糙度)
                     ↓
              13波段植被指数(含缨帽变换)
                     ↓
         28波段特征栈构建 + 4重掩膜
         (CLCD森林 + 质量掩膜 + 坡度 + NDVI)
                     ↓
         全部28维特征(不做特征筛选)
                     ↓
    2019年: PKU-AGB → Jenks 3级标签(含边界模糊过滤) → 分层空间采样 → HC/RF训练
    其他年: ────────复用2019模型────────→ 预测
```

### 28维特征空间

| 类别 | 特征 | 数量 |
|------|------|------|
| Landsat光谱 | B1(沿海气溶胶), B2(蓝), B3(绿), B4(红), B5(近红外), B6(SWIR1), B7(SWIR2) | 7 |
| 植被指数 | NDVI, EVI, NDMI, SAVI, NDWI, MSAVI, NBR, SR_B5B4, SR_B6B5, SR_B7B5, TCB, TCG, TCW | 13 |
| 地形因子 | elevation, slope, aspect, twi, pcurv, ccurv, slope_pos, roughness | 8 |

## 四、模型细节

### 分类器选择

- **主分类器**：HierarchicalClassifier (分层二分类)
  - L1分类器：Level1 vs (Level2+Level3)
  - L2分类器：Level2 vs Level3
  - 联合概率估计
- **备选**：RandomForestClassifier (scikit-learn)

### 训练参数

- **RF参数**：n_estimators=300, max_depth=30, min_samples_split=10, min_samples_leaf=8, max_features=0.7, class_weight="balanced"
- **分级**：Jenks Natural Breaks 3级 (低碳/中碳/高碳密度)
- **采样**：分层等量(per_class_target=2000) + 网格空间去自相关(min_spacing=5像素)
- **质量筛选**：slope<=30°, NDVI>=0.3 (宁缺毋滥策略)
- **样本容错**：允许最多30% NaN特征
- **边界过滤**：Jenks断点±8 t/ha范围样本剔除
- **低质量过滤**：所有特征与AGB相关性<0.1的像元剔除
- **验证**：5-fold CV + 30%独立验证集

### 模型性能（参考值）

| 研究区 | 分类器 | OA | Kappa | 特征数 | 训练/验证样本 |
|--------|--------|-----|-------|--------|--------------|
| 宁洱 | HierarchicalClassifier | 0.579 | 0.369 | 28 | 4200/1800 |
| 双柏 | HierarchicalClassifier | 0.694 | 0.542 | 28 | 4200/1800 |

## 五、研究区配置

两个研究区均位于云南省，使用 EPSG:32647 (UTM Zone 47N)，30m 分辨率：

| 研究区 | 位置 | DEM来源 | AGB来源 | CLCD来源 |
|--------|------|---------|---------|---------|
| 宁洱县 (ninger) | 普洱市 | 云南省DEM镶嵌(1.1GB) | AGB_2019_ninger.tif (PKU) | CLCD_v01_{year}_albert.tif |
| 双柏县 (shuangbai) | 楚雄州 | 同上 | AGB_2019_shuangbai.tif (PKU) | 同上 |

AGB数据来源：北京大学遥感与地理信息系统研究所 (https://www.irsgis.pku.edu.cn/xwdt/158207.htm)

## 六、核心依赖

| 类别 | 包 | 用途 |
|------|-----|------|
| 栅格处理 | rasterio>=1.3.0 | GeoTIFF读写、重投影、裁剪 |
| 矢量处理 | geopandas>=0.12.0, shapely>=2.0.0 | GeoJSON/Shapefile处理 |
| 数值计算 | numpy>=1.22.0, scipy>=1.9.0 | 数组运算、梯度计算 |
| 机器学习 | scikit-learn>=1.2.0, joblib>=1.2.0 | RF分类器、模型序列化 |
| 分级 | jenkspy>=2.0.0 | Jenks自然断点 |
| GUI | customtkinter>=5.2.0, Pillow>=9.0.0 | 桌面应用 |
| 卫星数据 | usgsxplore>=1.1.0, earthaccess>=0.5.0, h5py>=3.8.0 | Landsat/GEDI下载 |
| Web | fastapi>=0.100.0, uvicorn>=0.23.0 | 后端API |
| AI | openai>=1.0.0, sse-starlette>=1.6.0 | DeepSeek聊天 |

## 七、文件结构

```
forest_carbon_tierin/
├── config/                    # 研究区配置 (YAML) + API凭据
│   ├── ninger.yaml
│   ├── shuangbai.yaml
│   └── credentials/           # USGS/EarthData/DeepSeek凭据
├── src/                       # 核心处理流水线 (8步)
│   ├── config.py              # RegionConfig统一配置
│   ├── run_all.py             # 一键运行
│   ├── 01-07_*.py             # 各步骤脚本
│   └── 06b_hierarchical_classifier.py  # 分层二分类器
├── gui/                       # CustomTkinter桌面GUI
│   ├── app.py                 # 主窗口(5-Tab + 延迟加载)
│   ├── styles.py              # 浅色主题配色
│   ├── prediction_tab.py      # 碳汇预测(矢量边界裁剪 + 汉化工具栏)
│   ├── landsat_tab.py         # Landsat下载
│   ├── geojson_tab.py         # GeoJSON转换
│   ├── gedi_tab.py            # GEDI下载+处理
│   └── web_service_tab.py     # Web服务管理
├── web/                       # Web平台
│   ├── backend/
│   │   ├── main.py            # FastAPI主入口
│   │   ├── routers/chat.py    # DeepSeek AI聊天(SSE流式)
│   │   └── data/              # 预处理后Web展示数据
│   ├── frontend/
│   │   ├── index.html         # Vue3+ElementPlus+Leaflet+ECharts
│   │   └── libs/              # 本地化前端库
│   └── scripts/
│       └── generate_web_data.py  # 5步数据预处理
├── data/                      # 数据目录 (gitignore)
│   ├── shared/                # 共享数据(DEM/CLCD/缓存)
│   ├── ninger/                # 宁洱县数据(raw/intermediate/aligned)
│   └── shuangbai/             # 双柏县数据
└── output/                    # 输出目录 (gitignore)
    └── {region}/{year}/
        ├── metrics/           # hc_model.pkl/rf_model.pkl + 模型指标
        ├── figures/           # 分类图/混淆矩阵/特征重要性 PNG
        └── classification/    # carbon_density_class.tif
```

## 八、关键特性

1. **8步自动化流水线**：从原始卫星数据到分级制图全自动化，支持增量执行
2. **分层二分类器**：HierarchicalClassifier先分L1 vs L23，再分L2 vs L3，比直接三分类更合理
3. **28维特征空间**：7光谱+13植被指数+8地形因子，不做特征筛选，使用全部特征
4. **边界模糊过滤**：Jenks断点±8 t/ha范围样本剔除，减少边界等级混淆
5. **低质量样本过滤**：全局特征-AGB相关性<0.1的像元剔除
6. **WarpedVRT加速**：批量栅格对齐替代逐波段reproject
7. **跨年份预测**：2019年训练模型用于2023/2024/2025年，地形数据跨年复用
8. **PKU-AGB数据源**：使用北京大学发布的2019年森林AGB数据
9. **CLCD森林掩膜**：基于中国土地覆盖数据集自动提取森林区域，带缓存机制
10. **4重质量筛选**：森林掩膜 + 质量掩膜 + 坡度(≤30°) + NDVI(≥0.3)
11. **多接口访问**：命令行、GUI、Web三种使用方式
12. **Web时空监测**：Leaflet地图 + 天地图底图 + 分屏对比 + 点查询
13. **AI智能问答**：DeepSeek大模型 + 项目数据上下文注入，SSE流式对话
14. **置信度分析**：基于predict_proba生成预测置信度图，主动展示不确定性
15. **变化检测**：自动计算年份间碳汇等级变化(升级/降级/不变)，生成变化图和转移矩阵

## 九、使用方式

```bash
# 安装依赖
pip install -r requirements.txt

# 启动GUI
python -m gui.app   # 或双击 start.bat

# 命令行运行流水线
python -m src.run_all --region ninger --year 2019   # 完整流程(训练)
python -m src.run_all --region ninger --year 2023   # 预测模式(使用2019模型)

# 启动Web平台
python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload
# 或双击 start_web.bat
```

## 十、License

MIT License
