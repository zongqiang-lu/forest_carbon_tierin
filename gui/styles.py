"""
共享主题与样式定义
==================
统一管理 CustomTkinter 浅色主题配色、字体、常量
参考 VS Code / Notion / Figma 的简洁专业风格
"""

import customtkinter as ctk

# ---- 主题模式 ----
APPEARANCE_MODE = "light"
COLOR_THEME = "blue"

# ---- 浅色配色方案 (专业大气) ----
BG_PRIMARY = "#ffffff"       # 主背景 (纯白, 干净)
BG_SECONDARY = "#f8fafc"    # 次背景 (卡片区域, 极浅灰)
BG_TERTIARY = "#f1f5f9"     # 三级背景 (工具栏/分割区)
BG_INPUT = "#ffffff"         # 输入框背景
BG_CODE = "#0f172a"          # 代码/日志背景 (深蓝黑, 专业感)

FG_PRIMARY = "#0f172a"       # 主文字 (近黑)
FG_SECONDARY = "#475569"     # 次要文字 (深灰)
FG_MUTED = "#94a3b8"         # 弱化文字

ACCENT = "#16a34a"           # 主强调色 (绿)
ACCENT_HOVER = "#15803d"     # 强调色悬停
ACCENT_LIGHT = "#dcfce7"     # 强调色浅底
ACCENT_DIM = "#bbf7d0"       # 强调色淡底

DANGER = "#ef4444"           # 危险/错误
DANGER_LIGHT = "#fef2f2"     # 危险浅底
WARNING = "#f59e0b"          # 警告
WARNING_LIGHT = "#fffbeb"    # 警告浅底
INFO = "#3b82f6"             # 信息
INFO_LIGHT = "#eff6ff"       # 信息浅底

BORDER = "#e2e8f0"           # 边框
BORDER_FOCUS = "#16a34a"     # 聚焦边框

# ---- Tab标签专用颜色 ----
TAB_BG = "#f1f5f9"                    # Tab栏背景 (浅灰)
TAB_SELECTED_BG = "#ffffff"           # 选中Tab背景 (白)
TAB_SELECTED_TEXT = "#0f172a"         # 选中Tab文字 (深色, 醒目)
TAB_UNSELECTED_TEXT = "#64748b"       # 未选中Tab文字 (中灰, 清晰可读)
TAB_HOVER_TEXT = "#334155"            # 未选中Tab悬停文字

CARD_SHADOW = "#0f172a0a"    # 卡片阴影

# ---- 字体 (tuple格式) ----
FONT_FAMILY = "Microsoft YaHei UI"
FONT_TITLE = (FONT_FAMILY, 20, "bold")
FONT_SUBTITLE = (FONT_FAMILY, 15, "bold")
FONT_BODY = (FONT_FAMILY, 13)
FONT_SMALL = (FONT_FAMILY, 11)
FONT_MONO = ("Consolas", 11)

# ---- 分类颜色 (matplotlib) ----
CLASS_COLORS_RGBA = {
    1: (239, 68, 68, 255),     # Low - 红
    2: (245, 158, 11, 255),    # Medium - 琥珀
    3: (22, 163, 74, 255),     # High - 绿
}

CLASS_LABELS = {
    1: "低碳密度 (Low)",
    2: "中碳密度 (Medium)",
    3: "高碳密度 (High)",
}

# ---- 布局常量 ----
PADDING = 20
PADDING_SM = 10
PADDING_XS = 5
CORNER_RADIUS = 10
CARD_RADIUS = 12
WIDGET_HEIGHT = 36
BUTTON_HEIGHT = 40
PROGRESS_HEIGHT = 6

# ---- 项目路径 ----
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_DIR = PROJECT_ROOT / "config" / "credentials"

# ---- matplotlib 配色 (浅色模式) ----
MPL_FIGURE_BG = "#ffffff"
MPL_AXES_BG = "#f8fafc"
MPL_TEXT = "#0f172a"
MPL_SPINE = "#e2e8f0"
