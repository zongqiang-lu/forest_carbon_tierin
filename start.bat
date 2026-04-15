@echo off
chcp 65001 >nul
echo ========================================
echo   树智碳汇 - 启动GUI
echo ========================================
echo.

cd /d "%~dp0"
python -m gui.app
