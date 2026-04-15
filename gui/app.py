"""
树智碳汇 - 主入口
==============================
CustomTkinter 统一GUI，Tab布局集成五大功能模块
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import customtkinter as ctk
from gui.styles import (
    APPEARANCE_MODE, COLOR_THEME,
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, FG_PRIMARY, FG_SECONDARY,
    ACCENT, ACCENT_HOVER, ACCENT_LIGHT,
    TAB_BG, TAB_SELECTED_BG, TAB_SELECTED_TEXT, TAB_UNSELECTED_TEXT, TAB_HOVER_TEXT,
    FONT_TITLE, FONT_BODY, FONT_SMALL, FONT_SUBTITLE,
    PADDING, PADDING_SM, PADDING_XS, PROJECT_ROOT, BORDER, CARD_RADIUS,
)


class CarbonTierinApp(ctk.CTk):
    """主应用窗口"""

    def __init__(self):
        super().__init__()

        # 主题设置
        ctk.set_appearance_mode(APPEARANCE_MODE)
        ctk.set_default_color_theme(COLOR_THEME)

        # 窗口配置
        self.title("树智碳汇")
        self.geometry("1360x900")
        self.minsize(1100, 750)
        self.configure(fg_color=BG_PRIMARY)

        self._build_header()
        self._build_tabview()
        self._build_statusbar()

    def _build_header(self):
        """顶部标题栏 - 大气专业"""
        header = ctk.CTkFrame(self, height=64, fg_color="#ffffff", corner_radius=0,
                              border_width=0)
        header.pack(fill="x")
        header.pack_propagate(False)

        # 左侧Logo和标题
        left = ctk.CTkFrame(header, fg_color="transparent")
        left.pack(side="left", padx=PADDING, fill="y")

        ctk.CTkLabel(
            left, text="🌲",
            font=("", 24),
        ).pack(side="left", padx=(0, 10), pady=14)

        title_col = ctk.CTkFrame(left, fg_color="transparent")
        title_col.pack(side="left", fill="y")

        ctk.CTkLabel(
            title_col, text="树智碳汇",
            font=(FONT_TITLE[0], 17, "bold"), text_color=FG_PRIMARY,
        ).pack(anchor="w", pady=(12, 0))

        ctk.CTkLabel(
            title_col, text="TreeWit Carbon - Forest Carbon Sink Intelligence Platform",
            font=(FONT_SMALL[0], 10), text_color=FG_SECONDARY,
        ).pack(anchor="w")

        # 右侧版本号
        ctk.CTkLabel(
            header, text="v1.0",
            font=FONT_SMALL, text_color=FG_SECONDARY,
        ).pack(side="right", padx=PADDING, pady=14)

        # 底部分割线
        ctk.CTkFrame(self, height=1, fg_color=BORDER, corner_radius=0).pack(fill="x")

    def _build_tabview(self):
        """主体TabView - 自定义Tab样式使标签更醒目"""
        # Tab容器
        tab_container = ctk.CTkFrame(self, fg_color=BG_PRIMARY, corner_radius=0)
        tab_container.pack(fill="both", expand=True, padx=0, pady=0)

        self.tabview = ctk.CTkTabview(
            tab_container,
            fg_color=BG_PRIMARY,
            segmented_button_fg_color=TAB_BG,
            segmented_button_selected_color=TAB_SELECTED_BG,
            segmented_button_selected_hover_color="#f8fafc",
            segmented_button_unselected_color=TAB_BG,
            segmented_button_unselected_hover_color="#e8edf3",
            corner_radius=CARD_RADIUS,
            border_width=0,
        )
        self.tabview.pack(fill="both", expand=True, padx=PADDING, pady=(PADDING_SM, 0))

        tab_names = ["碳汇预测", "Landsat下载", "GeoJSON转换", "GEDI下载", "Web服务"]
        self.tabs = {}
        for name in tab_names:
            self.tabs[name] = self.tabview.add(name)

        # 修正Tab标签文字颜色 - 使其更清晰
        self._fix_tab_text_colors()

        self._load_tabs()

        # 监听Tab切换事件 - 用于延迟加载
        # 保存原始command，先执行原始切换逻辑，再触发延迟初始化
        self._orig_seg_cmd = self.tabview._segmented_button.cget("command")
        self.tabview._segmented_button.configure(command=self._on_tab_changed)

        # 首次加载后触发当前Tab的延迟初始化（默认是碳汇预测）
        self.after(100, self._trigger_initial_tab)

    def _fix_tab_text_colors(self):
        """修正CTkTabview标签文字颜色，确保清晰可读"""
        try:
            seg_btn = self.tabview._segmented_button
            # 设置选中标签文字颜色 (深色)
            seg_btn.configure(
                text_color=TAB_SELECTED_TEXT,
                text_color_disabled=TAB_UNSELECTED_TEXT,
            )
            # 遍历所有按钮设置颜色
            for i, btn in enumerate(seg_btn._button_dict.values()):
                btn.configure(
                    font=(FONT_BODY[0], 13, "bold"),
                )
        except Exception:
            pass

    def _on_tab_changed(self, tab_name):
        """Tab切换回调 - 先执行原始切换逻辑，再触发延迟初始化"""
        # 调用原始的Tab切换命令
        if self._orig_seg_cmd:
            self._orig_seg_cmd(tab_name)
        # 触发延迟初始化
        tab_instance = self._tab_instances.get(tab_name)
        if tab_instance and hasattr(tab_instance, "ensure_initialized"):
            tab_instance.ensure_initialized()

    def _trigger_initial_tab(self):
        """触发默认Tab的延迟初始化"""
        try:
            current = self.tabview.get()
            if current:
                self._on_tab_changed(current)
        except Exception:
            pass

    def _load_tabs(self):
        """加载各Tab页内容"""
        self._tab_instances = {}
        tab_loaders = [
            ("碳汇预测", "gui.prediction_tab", "PredictionTab"),
            ("Landsat下载", "gui.landsat_tab", "LandsatTab"),
            ("GeoJSON转换", "gui.geojson_tab", "GeoJSONTab"),
            ("GEDI下载", "gui.gedi_tab", "GEDITab"),
            ("Web服务", "gui.web_service_tab", "WebServiceTab"),
        ]
        for tab_name, module_name, class_name in tab_loaders:
            try:
                mod = __import__(module_name, fromlist=[class_name])
                cls = getattr(mod, class_name)
                instance = cls(self.tabs[tab_name], self)
                self._tab_instances[tab_name] = instance
            except Exception as e:
                import traceback
                traceback.print_exc()
                err_label = ctk.CTkLabel(
                    self.tabs[tab_name],
                    text=f"加载{tab_name}模块失败:\n{e}",
                    text_color="#ef4444", font=FONT_BODY,
                )
                err_label.pack(padx=PADDING, pady=PADDING, anchor="nw")

    def _build_statusbar(self):
        """底部状态栏"""
        # 顶部分割线
        ctk.CTkFrame(self, height=1, fg_color=BORDER, corner_radius=0).pack(fill="x", side="bottom")

        self.statusbar = ctk.CTkFrame(self, height=32, fg_color="#ffffff", corner_radius=0)
        self.statusbar.pack(fill="x", side="bottom")
        self.statusbar.pack_propagate(False)

        self.status_var = ctk.StringVar(value="就绪")
        ctk.CTkLabel(
            self.statusbar, textvariable=self.status_var,
            font=FONT_SMALL, text_color=FG_SECONDARY, anchor="w",
        ).pack(fill="x", padx=PADDING, pady=4)

    def set_status(self, msg: str):
        """更新状态栏"""
        self.status_var.set(msg)


def main():
    app = CarbonTierinApp()
    app.mainloop()


if __name__ == "__main__":
    main()
