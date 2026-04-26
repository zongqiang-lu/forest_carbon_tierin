# .gitignore 忽略文件详情（具体文件列表）
## 统计信息
总忽略文件数：386个
总大小：约45GB

## 分类列表
### 一、敏感凭据（3个）
| 文件路径 | 说明 | 忽略原因 |
|---------|------|---------|
| config/credentials/deepseek.json | DeepSeek API密钥 | 敏感信息，禁止公开 |
| config/credentials/earthdata.json | NASA EarthData登录凭据 | 敏感信息 |
| config/credentials/usgs.json | USGS API访问密钥 | 敏感信息 |

### 二、Python缓存文件（26个）
所有`__pycache__`目录下的`.pyc`字节码文件，运行时自动生成，无需版本控制，共26个，分布在：
- `gui/__pycache__/` 8个
- `src/__pycache__/` 2个
- `web/__pycache__/` 1个
- `web/backend/__pycache__/` 3个
- `web/backend/routers/__pycache__/` 2个
- `web/scripts/__pycache__/` 2个

### 三、Landsat原始影像（216个）
美国地质调查局下载的Landsat 8/9卫星原始影像，每个影像包含11个波段文件+元数据文件，覆盖巍山、双柏两个研究区2019/2023/2024/2025四个年份，共216个：
- 位置：`data/[研究区]/[年份]/raw/landsat/[影像ID]/`
- 文件类型：`.TIF`（影像波段）、`.txt`（元数据）、`.xml`（元数据）
- 总大小：约32GB

### 四、中间处理数据（75个）
影像预处理、特征工程生成的中间文件：
- 位置：`data/[研究区]/[年份]/intermediate/`、`data/[研究区]/[年份]/aligned/`、`data/shared/clcd_cache/`
- 文件类型：`.tif`（栅格数据）、`.json`（元数据）
- 包含内容：NDVI等植被指数、地形特征、质量掩码、特征堆栈、森林掩码等

### 五、基础地理数据（16个）
公开地理数据集：
- 位置：`data/[研究区]/raw/agb/`、`data/[研究区]/raw/boundary/`、`data/shared/clcd/`、`data/shared/dem/`
- 包含内容：地面实测AGB数据、行政区边界、CLCD土地覆盖数据、DEM高程数据

### 六、模型输出结果（37个）
随机森林模型训练、预测生成的结果文件：
- 位置：`output/[研究区]/[年份]/`
- 包含内容：分类结果栅格、模型指标、特征重要性、可视化图片、训练好的模型文件(.pkl)

### 七、Web服务生成数据（36个）
Web前端可视化使用的预处理数据：
- 位置：`web/backend/data/`
- 包含内容：边界GeoJSON、动态变化栅格、置信度图层、切图数据、统计JSON文件
