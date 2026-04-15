@echo off
chcp 65001 >nul
echo ========================================
echo   树智碳汇 - 启动Web平台
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] 检查依赖...
pip show fastapi >nul 2>&1 || (
    echo 安装Web依赖...
    pip install fastapi uvicorn python-multipart Pillow -q
)

echo [2/2] 检查预处理数据...
if not exist "web\backend\data\stats\all_stats.json" (
    echo.
    echo ⚠️  未检测到预处理数据，正在生成...
    echo    这可能需要几分钟时间...
    python -m web.scripts.generate_web_data
)

echo.
echo 启动后端服务 (http://localhost:8000)...
echo API文档: http://localhost:8000/docs
echo.

start "" http://localhost:8000/app

python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload
