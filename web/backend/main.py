"""
FastAPI后端 - 主入口
====================
提供地图数据、统计分析、点查询、报表导出、AI智能问答等API
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from pydantic import BaseModel
from typing import Optional
import csv
import io

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WEB_DATA_DIR = Path(__file__).resolve().parent / "data"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="树智碳汇 API",
    description="提供碳汇时空动态监测、统计分析、不确定性分析等功能",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 AI 聊天路由
from web.backend.routers.chat import router as chat_router
app.include_router(chat_router)

# 挂载静态文件 - 地图PNG/瓦片
if (WEB_DATA_DIR / "maps").exists():
    app.mount("/data/maps", StaticFiles(directory=str(WEB_DATA_DIR / "maps")), name="maps")
if (WEB_DATA_DIR / "change").exists():
    app.mount("/data/change", StaticFiles(directory=str(WEB_DATA_DIR / "change")), name="change")
if (WEB_DATA_DIR / "confidence").exists():
    app.mount("/data/confidence", StaticFiles(directory=str(WEB_DATA_DIR / "confidence")), name="confidence")
if (WEB_DATA_DIR / "boundaries").exists():
    app.mount("/data/boundaries", StaticFiles(directory=str(WEB_DATA_DIR / "boundaries")), name="boundaries")

# 挂载前端库文件 (本地化, 避免CDN慢)
LIBS_DIR = FRONTEND_DIR / "libs"
if LIBS_DIR.exists():
    app.mount("/libs", StaticFiles(directory=str(LIBS_DIR)), name="libs")


# 前端页面
@app.get("/app")
async def serve_frontend():
    """提供前端页面"""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return JSONResponse({"error": "前端文件未找到，请确认 web/frontend/index.html 存在"}, status_code=404)


# ============ 数据模型 ============

class PointQuery(BaseModel):
    lng: float
    lat: float
    region: str
    year: int


class RegionInfo(BaseModel):
    region_id: str
    region_name: str
    bounds: dict


# ============ 辅助函数 ============

def load_json(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"数据文件不存在: {path.name}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============ API路由 ============

@app.get("/")
async def root():
    return {"message": "树智碳汇 API", "version": "1.0.0"}


# --- 基础信息 ---

@app.get("/api/regions")
async def get_regions():
    """获取所有研究区信息"""
    regions = []
    for region_id in ["ninger", "shuangbai"]:
        name = "宁洱县" if region_id == "ninger" else "双柏县"
        # 读取bounds
        bounds_path = WEB_DATA_DIR / "maps" / region_id / "2019" / f"{region_id}_2019_bounds.json"
        bounds = load_json(bounds_path) if bounds_path.exists() else {}
        regions.append({
            "region_id": region_id,
            "region_name": name,
            "bounds": bounds,
            "years": [2019, 2023, 2024, 2025]
        })
    return regions


@app.get("/api/map/bounds/{region}/{year}")
async def get_map_bounds(region: str, year: int):
    """获取地图边界"""
    bounds_path = WEB_DATA_DIR / "maps" / region / str(year) / f"{region}_{year}_bounds.json"
    return load_json(bounds_path)


@app.get("/api/map/image/{region}/{year}")
async def get_map_image(region: str, year: int):
    """获取分类图PNG"""
    img_path = WEB_DATA_DIR / "maps" / region / str(year) / f"{region}_{year}.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="地图图片不存在")
    return FileResponse(str(img_path), media_type="image/png")


@app.get("/api/map/change/{region}/{year1}/{year2}")
async def get_change_map(region: str, year1: int, year2: int):
    """获取变化检测图"""
    img_path = WEB_DATA_DIR / "change" / region / f"change_{year1}_{year2}.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="变化图不存在")
    return FileResponse(str(img_path), media_type="image/png")


@app.get("/api/map/change_bounds/{region}/{year1}/{year2}")
async def get_change_bounds(region: str, year1: int, year2: int):
    """获取变化图边界"""
    bounds_path = WEB_DATA_DIR / "change" / region / f"change_{year1}_{year2}_bounds.json"
    return load_json(bounds_path)


@app.get("/api/map/confidence/{region}")
async def get_confidence_map(region: str):
    """获取置信度图"""
    # 兼容两种命名: _2019 (当前) 和 _2023 (旧版)
    img_path = WEB_DATA_DIR / "confidence" / region / f"confidence_{region}_2019.png"
    if not img_path.exists():
        img_path = WEB_DATA_DIR / "confidence" / region / f"confidence_{region}_2023.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="置信度图不存在")
    return FileResponse(str(img_path), media_type="image/png")


@app.get("/api/map/confidence_bounds/{region}")
async def get_confidence_bounds(region: str):
    """获取置信度图边界"""
    bounds_path = WEB_DATA_DIR / "confidence" / region / f"confidence_{region}_2019_bounds.json"
    if not bounds_path.exists():
        bounds_path = WEB_DATA_DIR / "confidence" / region / f"confidence_{region}_2023_bounds.json"
    return load_json(bounds_path)


# --- 边界 ---

@app.get("/api/boundary/{region}")
async def get_boundary(region: str):
    """获取研究区边界GeoJSON"""
    geojson_path = WEB_DATA_DIR / "boundaries" / f"{region}.geojson"
    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail="边界数据不存在")
    return load_json(geojson_path)


# --- 统计 ---

@app.get("/api/stats/overview")
async def get_stats_overview():
    """获取统计概览"""
    stats_path = WEB_DATA_DIR / "stats" / "all_stats.json"
    return load_json(stats_path)


@app.get("/api/stats/changes")
async def get_changes():
    """获取变化检测统计"""
    changes_path = WEB_DATA_DIR / "stats" / "all_changes.json"
    return load_json(changes_path)


@app.get("/api/stats/region/{region}")
async def get_region_stats(region: str, year: Optional[int] = None):
    """获取某区域统计（可指定年份）"""
    all_stats = load_json(WEB_DATA_DIR / "stats" / "all_stats.json")
    region_data = all_stats.get(region, [])
    if year:
        region_data = [s for s in region_data if s["year"] == year]
    return region_data


@app.get("/api/stats/confidence/{region}")
async def get_confidence_stats(region: str):
    """获取置信度统计"""
    stats_path = WEB_DATA_DIR / "confidence" / region / f"confidence_stats_{region}.json"
    return load_json(stats_path)


# --- 模型指标 ---

@app.get("/api/model/metrics/{region}")
async def get_model_metrics(region: str):
    """获取模型评估指标"""
    metrics_path = WEB_DATA_DIR / "stats" / f"model_metrics_{region}.json"
    return load_json(metrics_path)


@app.get("/api/model/feature_importance/{region}")
async def get_feature_importance(region: str):
    """获取特征重要性"""
    csv_path = WEB_DATA_DIR / "stats" / f"feature_importance_{region}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="特征重要性数据不存在")
    
    features = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            features.append({
                "feature": row["feature"],
                "importance": float(row["importance"])
            })
    return features


@app.get("/api/model/breaks/{region}")
async def get_classification_breaks(region: str):
    """获取分类断点"""
    breaks_path = WEB_DATA_DIR / "stats" / f"jenks_breaks_{region}.json"
    return load_json(breaks_path)


# --- 点查询 ---

@app.post("/api/query/point")
async def query_point(query: PointQuery):
    """
    查询某点的详细信息
    包括：碳汇等级、特征值、置信度、多年度对比
    """
    result = {
        "lng": query.lng,
        "lat": query.lat,
        "region": query.region,
        "year": query.year,
        "carbon_class": None,
        "features": {},
        "multi_year": {}
    }
    
    # 1. 查询分类等级
    tif_path = WEB_DATA_DIR / "maps" / query.region / str(query.year) / f"{query.region}_{query.year}_wgs84.tif"
    if not tif_path.exists():
        tif_path = PROJECT_ROOT / f"output/{query.region}/{query.year}/classification/carbon_density_class.tif"
    
    if tif_path.exists():
        try:
            with rasterio.open(tif_path) as src:
                if src.crs and src.crs.to_epsg() != 4326:
                    # 需要坐标转换
                    xs, ys = warp_transform("EPSG:4326", src.crs, [query.lng], [query.lat])
                    row, col = src.index(xs[0], ys[0])
                else:
                    row, col = src.index(query.lng, query.lat)
                
                if 0 <= row < src.height and 0 <= col < src.width:
                    val = int(src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0])
                    result["carbon_class"] = val if val > 0 else None
        except Exception:
            pass
    
    # 2. 查询特征值 (从feature_stack)
    feature_stack_path = PROJECT_ROOT / f"data/{query.region}/{query.year}/aligned/feature_stack.tif"
    feature_names_path = PROJECT_ROOT / f"data/{query.region}/{query.year}/aligned/feature_names.json"
    
    if feature_stack_path.exists() and feature_names_path.exists():
        try:
            with open(feature_names_path, 'r') as f:
                fnames = json.load(f)["feature_names"]
            
            with rasterio.open(feature_stack_path) as src:
                if src.crs and src.crs.to_epsg() != 4326:
                    xs, ys = warp_transform("EPSG:4326", src.crs, [query.lng], [query.lat])
                    row, col = src.index(xs[0], ys[0])
                else:
                    row, col = src.index(query.lng, query.lat)
                
                if 0 <= row < src.height and 0 <= col < src.width:
                    values = src.read(window=rasterio.windows.Window(col, row, 1, 1)).flatten()
                    for i, name in enumerate(fnames):
                        if i < len(values):
                            v = float(values[i])
                            if not (np.isnan(v) or np.isinf(v)):
                                result["features"][name] = round(v, 4)
        except Exception:
            pass
    
    # 3. 多年度对比
    for year in [2019, 2023, 2024, 2025]:
        year_tif = PROJECT_ROOT / f"output/{query.region}/{year}/classification/carbon_density_class.tif"
        if year_tif.exists():
            try:
                with rasterio.open(year_tif) as src:
                    if src.crs and src.crs.to_epsg() != 4326:
                        xs, ys = warp_transform("EPSG:4326", src.crs, [query.lng], [query.lat])
                        row, col = src.index(xs[0], ys[0])
                    else:
                        row, col = src.index(query.lng, query.lat)
                    
                    if 0 <= row < src.height and 0 <= col < src.width:
                        val = int(src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0])
                        result["multi_year"][str(year)] = val if val > 0 else None
            except Exception:
                pass
    
    # 4. 置信度
    conf_path = WEB_DATA_DIR / "confidence" / query.region / f"confidence_{query.region}_2019_wgs84.tif"
    if conf_path.exists():
        try:
            with rasterio.open(conf_path) as src:
                row, col = src.index(query.lng, query.lat)
                if 0 <= row < src.height and 0 <= col < src.width:
                    val = float(src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0])
                    result["confidence"] = round(val, 4) if val > 0 else None
        except Exception:
            pass
    
    return result


# --- 图表 ---

@app.get("/api/figures/{region}/{year}/{name}")
async def get_figure(region: str, year: int, name: str):
    """获取可视化图表"""
    fig_path = PROJECT_ROOT / f"output/{region}/{year}/figures/{name}"
    if not fig_path.exists():
        raise HTTPException(status_code=404, detail="图表不存在")
    media_type = "image/png"
    return FileResponse(str(fig_path), media_type=media_type)


# --- 导出 ---

@app.get("/api/export/stats/{region}")
async def export_stats(region: str, format: str = Query("json", regex="^(json|csv)$")):
    """导出统计数据"""
    all_stats = load_json(WEB_DATA_DIR / "stats" / "all_stats.json")
    region_data = all_stats.get(region, [])
    
    if format == "csv":
        output = io.StringIO()
        if region_data:
            writer = csv.DictWriter(output, fieldnames=region_data[0].keys())
            writer.writeheader()
            writer.writerows(region_data)
        return JSONResponse(
            content={"csv": output.getvalue()},
            media_type="application/json"
        )
    return region_data


# --- 健康检查 ---

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "data_dir": str(WEB_DATA_DIR), "data_exists": WEB_DATA_DIR.exists()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
