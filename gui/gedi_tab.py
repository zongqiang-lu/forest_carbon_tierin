"""
GEDI L4A 数据下载与处理Tab
============================
从gedi_download/gedi_l4a_gui.py重构为CTkFrame可嵌入组件
"""

import os
import json
import threading
import numpy as np
import pandas as pd

import customtkinter as ctk
from tkinter import filedialog, messagebox

from gui.styles import (
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BG_INPUT, BG_CODE,
    FG_PRIMARY, FG_SECONDARY, FG_MUTED,
    ACCENT, ACCENT_HOVER, ACCENT_LIGHT, ACCENT_DIM,
    DANGER, WARNING, INFO, INFO_LIGHT,
    FONT_SUBTITLE, FONT_BODY, FONT_SMALL, FONT_MONO,
    PADDING, PADDING_SM, PADDING_XS, CARD_RADIUS, CREDENTIALS_DIR, BORDER, PROGRESS_HEIGHT,
)

CREDENTIALS_FILE = CREDENTIALS_DIR / "earthdata.json"
GEDI_L4A_COLLECTION_ID = "C2237824918-ORNL_CLOUD"

BEAMS = [
    "BEAM0000", "BEAM0001", "BEAM0010", "BEAM0011",
    "BEAM0100", "BEAM0101", "BEAM0110", "BEAM0111",
]

VARIABLES = {
    "lon_lowestmode": "lon", "lat_lowestmode": "lat",
    "agbd": "agbd", "agbd_se": "agbd_se",
    "elev_lowestmode": "elev", "sensitivity": "sensitivity",
    "l2_quality_flag": "l2_quality_flag", "l4_quality_flag": "l4_quality_flag",
    "degrade_flag": "degrade_flag", "delta_time": "delta_time",
    "shot_number": "shot_number",
}

try:
    from harmony import Client, Request, Collection, BBox, WKT
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False


def _load_credentials():
    if CREDENTIALS_FILE.exists():
        try:
            with open(CREDENTIALS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"earthdata_username": "", "earthdata_password": "", "default_download_dir": "./data", "default_output_file": "gedi_l4a.gpkg"}


def _save_credentials(config: dict):
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(CREDENTIALS_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception:
        pass


class GEDITab:
    """GEDI下载Tab页"""

    def __init__(self, parent: ctk.CTkFrame, app: ctk.CTk):
        self.parent = parent
        self.app = app
        self.config = _load_credentials()
        self.search_results = []
        self.is_downloading = False
        self.is_processing = False
        self._build_ui()

    def _make_card(self, parent, **kw):
        return ctk.CTkFrame(
            parent, fg_color=BG_SECONDARY, corner_radius=CARD_RADIUS,
            border_width=1, border_color=BORDER, **kw
        )

    def _make_entry(self, parent, **kw):
        return ctk.CTkEntry(
            parent, font=FONT_BODY, height=36,
            fg_color=BG_INPUT, text_color=FG_PRIMARY,
            border_color=BORDER, border_width=1, **kw
        )

    def _make_label(self, parent, text, **kw):
        return ctk.CTkLabel(
            parent, text=text, font=FONT_BODY, text_color=FG_SECONDARY,
            anchor="w", **kw
        )

    def _make_file_row(self, parent, label_text, row, variable, browse_cmd):
        """创建带浏览按钮的文件/目录选择行"""
        self._make_label(parent, label_text, width=140).grid(
            row=row, column=0, padx=(PADDING, PADDING_SM), pady=PADDING_SM, sticky="w")
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=1, padx=(PADDING_SM, PADDING), pady=PADDING_SM, sticky="ew")
        frame.grid_columnconfigure(0, weight=1)
        self._make_entry(frame, textvariable=variable).grid(row=0, column=0, sticky="ew", padx=(0, PADDING_XS))
        ctk.CTkButton(
            frame, text="浏览", width=60, height=36, font=FONT_BODY,
            fg_color=BG_TERTIARY, hover_color=BG_SECONDARY, text_color=FG_PRIMARY,
            command=browse_cmd,
        ).grid(row=0, column=1)

    def _build_ui(self):
        """构建界面 - 双子Tab"""
        self.main_frame = ctk.CTkFrame(self.parent, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=0, pady=0)

        self.tabview = ctk.CTkTabview(
            self.main_frame,
            segmented_button_fg_color=BG_TERTIARY,
            segmented_button_selected_color=BG_PRIMARY,
            segmented_button_selected_hover_color=BG_SECONDARY,
            segmented_button_unselected_color=BG_TERTIARY,
            corner_radius=CARD_RADIUS,
            border_width=1, border_color=BORDER,
        )
        self.tabview.pack(fill="both", expand=True, padx=PADDING, pady=PADDING)

        self.tab_download = self.tabview.add("数据下载")
        self.tab_process = self.tabview.add("数据处理")

        self._build_download_tab()
        self._build_process_tab()

    # ==================== 下载Tab ====================

    def _build_download_tab(self):
        tab = self.tab_download
        tab.grid_columnconfigure(1, weight=1)
        row = 0

        # 标题
        ctk.CTkLabel(tab, text="GEDI L4A 数据下载", font=FONT_SUBTITLE, text_color=ACCENT).grid(
            row=row, column=0, columnspan=2, padx=PADDING, pady=(PADDING, PADDING_SM), sticky="w")
        row += 1

        # 账号
        self._make_label(tab, "EarthData 用户名:", width=140).grid(
            row=row, column=0, padx=(PADDING, PADDING_SM), pady=PADDING_XS, sticky="w")
        self.username = ctk.StringVar(value=self.config.get("earthdata_username", ""))
        self._make_entry(tab, textvariable=self.username).grid(
            row=row, column=1, padx=(PADDING_SM, PADDING), pady=PADDING_XS, sticky="ew")
        row += 1

        # 密码
        self._make_label(tab, "EarthData 密码:", width=140).grid(
            row=row, column=0, padx=(PADDING, PADDING_SM), pady=PADDING_XS, sticky="w")
        self.password = ctk.StringVar(value=self.config.get("earthdata_password", ""))
        self._make_entry(tab, textvariable=self.password, show="*").grid(
            row=row, column=1, padx=(PADDING_SM, PADDING), pady=PADDING_XS, sticky="ew")
        row += 1

        # 记住凭据
        self.save_credentials_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            tab, text="记住凭据", variable=self.save_credentials_var, font=FONT_BODY,
            text_color=FG_PRIMARY, checkbox_width=20, checkbox_height=20,
        ).grid(row=row, column=0, columnspan=2, padx=PADDING, pady=PADDING_XS, sticky="w")
        row += 1

        # 下载方式
        self._make_label(tab, "下载方式:", width=140).grid(
            row=row, column=0, padx=(PADDING, PADDING_SM), pady=(PADDING_SM, PADDING_XS), sticky="w")
        method_frame = ctk.CTkFrame(tab, fg_color="transparent")
        method_frame.grid(row=row, column=1, padx=(PADDING_SM, PADDING), pady=(PADDING_SM, PADDING_XS), sticky="w")

        self.download_method = ctk.StringVar(value="harmony" if HARMONY_AVAILABLE else "earthaccess")
        ctk.CTkRadioButton(
            method_frame, text="完整轨道", variable=self.download_method, value="earthaccess",
            command=self._on_method_change, font=FONT_BODY, text_color=FG_PRIMARY,
        ).pack(side="left", padx=(0, PADDING))
        rb_harmony = ctk.CTkRadioButton(
            method_frame, text="空间子集化(推荐)", variable=self.download_method, value="harmony",
            command=self._on_method_change, font=FONT_BODY, text_color=FG_PRIMARY,
        )
        rb_harmony.pack(side="left")
        if not HARMONY_AVAILABLE:
            rb_harmony.configure(state="disabled")
        row += 1

        # AOI / 日期 / 目录
        self.aoi_file_download = ctk.StringVar()
        self._make_file_row(tab, "AOI文件(GeoJSON):", row, self.aoi_file_download, self._browse_aoi_download)
        row += 1

        self._make_label(tab, "开始日期:", width=140).grid(
            row=row, column=0, padx=(PADDING, PADDING_SM), pady=PADDING_XS, sticky="w")
        self.start_date = ctk.StringVar(value="2022-09-01")
        self._make_entry(tab, textvariable=self.start_date).grid(
            row=row, column=1, padx=(PADDING_SM, PADDING), pady=PADDING_XS, sticky="ew")
        row += 1

        self._make_label(tab, "结束日期:", width=140).grid(
            row=row, column=0, padx=(PADDING, PADDING_SM), pady=PADDING_XS, sticky="w")
        self.end_date = ctk.StringVar(value="2022-10-31")
        self._make_entry(tab, textvariable=self.end_date).grid(
            row=row, column=1, padx=(PADDING_SM, PADDING), pady=PADDING_XS, sticky="ew")
        row += 1

        self.download_dir = ctk.StringVar(value=self.config.get("default_download_dir", "./data"))
        self._make_file_row(tab, "下载目录:", row, self.download_dir, self._browse_download_dir)
        row += 1

        # 按钮
        btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
        btn_frame.grid(row=row, column=0, columnspan=2, padx=PADDING, pady=PADDING_SM, sticky="ew")
        btn_frame.grid_columnconfigure((0, 1), weight=1)
        self.search_btn = ctk.CTkButton(
            btn_frame, text="搜索数据", font=FONT_BODY, height=38,
            fg_color=BG_TERTIARY, hover_color=BG_SECONDARY, text_color=FG_PRIMARY,
            command=self._search_data,
        )
        self.search_btn.grid(row=0, column=0, padx=PADDING_XS, sticky="ew")
        self.download_btn = ctk.CTkButton(
            btn_frame, text="下载数据", font=FONT_BODY, height=38,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#ffffff",
            command=self._download_data, state="disabled",
        )
        self.download_btn.grid(row=0, column=1, padx=PADDING_XS, sticky="ew")
        row += 1

        # 进度条
        self.progress_download = ctk.CTkProgressBar(tab, height=PROGRESS_HEIGHT,
                                                     fg_color=BG_TERTIARY, progress_color=ACCENT)
        self.progress_download.grid(row=row, column=0, columnspan=2, padx=PADDING, pady=PADDING_XS, sticky="ew")
        self.progress_download.set(0)
        row += 1

        # 状态
        self.status_download = ctk.CTkLabel(tab, text="就绪", font=FONT_SMALL, text_color=FG_MUTED)
        self.status_download.grid(row=row, column=0, columnspan=2, padx=PADDING, pady=PADDING_XS)
        row += 1

        # 搜索结果
        self.results_frame = ctk.CTkScrollableFrame(tab, height=100, fg_color=BG_CODE, corner_radius=8)
        self.results_frame.grid(row=row, column=0, columnspan=2, padx=PADDING, pady=(PADDING_XS, PADDING), sticky="ew")
        ctk.CTkLabel(self.results_frame, text="尚未搜索", font=FONT_SMALL, text_color="#94a3b8").pack(pady=PADDING_SM)

        self._on_method_change()

    def _on_method_change(self):
        if self.download_method.get() == "harmony":
            if hasattr(self, "search_btn"):
                self.search_btn.configure(state="disabled")
            if hasattr(self, "download_btn"):
                self.download_btn.configure(state="normal")
        else:
            if hasattr(self, "search_btn"):
                self.search_btn.configure(state="normal")
            if hasattr(self, "download_btn"):
                self.download_btn.configure(state="disabled")

    # ==================== 处理Tab ====================

    def _build_process_tab(self):
        tab = self.tab_process
        tab.grid_columnconfigure(1, weight=1)
        row = 0

        ctk.CTkLabel(tab, text="GEDI L4A 数据处理", font=FONT_SUBTITLE, text_color=ACCENT).grid(
            row=row, column=0, columnspan=2, padx=PADDING, pady=(PADDING, PADDING_SM), sticky="w")
        row += 1

        self.data_dir = ctk.StringVar(value="./data")
        self._make_file_row(tab, "H5数据目录:", row, self.data_dir, self._browse_data_dir)
        row += 1

        self.aoi_file_process = ctk.StringVar()
        self._make_file_row(tab, "AOI文件(GeoJSON):", row, self.aoi_file_process, self._browse_aoi_process)
        row += 1

        self._make_label(tab, "输出文件名:", width=140).grid(
            row=row, column=0, padx=(PADDING, PADDING_SM), pady=PADDING_SM, sticky="w")
        self.output_file = ctk.StringVar(value=self.config.get("default_output_file", "gedi_l4a.gpkg"))
        self._make_entry(tab, textvariable=self.output_file).grid(
            row=row, column=1, padx=(PADDING_SM, PADDING), pady=PADDING_SM, sticky="ew")
        row += 1

        # 高级设置折叠
        self.advanced_visible = False
        self.advanced_toggle = ctk.CTkButton(
            tab, text="▶ 高级设置", width=120, font=FONT_SMALL,
            fg_color="transparent", text_color=FG_MUTED, hover_color=BG_TERTIARY,
            command=self._toggle_advanced,
        )
        self.advanced_toggle.grid(row=row, column=0, padx=PADDING, pady=PADDING_XS, sticky="w")
        row += 1

        self.advanced_frame = self._make_card(tab)
        self.advanced_frame.grid_columnconfigure(1, weight=1)
        self._build_advanced_settings()

        # 处理按钮
        self.process_btn = ctk.CTkButton(
            tab, text="开始处理", height=40, font=FONT_BODY,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#ffffff",
            command=self._run_process,
        )
        self.process_btn.grid(row=row, column=0, columnspan=2, padx=PADDING, pady=PADDING, sticky="ew")
        row += 1

        self.progress_process = ctk.CTkProgressBar(tab, height=PROGRESS_HEIGHT,
                                                     fg_color=BG_TERTIARY, progress_color=ACCENT)
        self.progress_process.grid(row=row, column=0, columnspan=2, padx=PADDING, pady=PADDING_XS, sticky="ew")
        self.progress_process.set(0)
        row += 1

        self.status_process = ctk.CTkLabel(tab, text="就绪", font=FONT_SMALL, text_color=FG_MUTED)
        self.status_process.grid(row=row, column=0, columnspan=2, padx=PADDING, pady=PADDING_XS)

    def _build_advanced_settings(self):
        f = self.advanced_frame
        row = 0
        ctk.CTkLabel(f, text="质量过滤设置", font=FONT_SUBTITLE, text_color=ACCENT).grid(
            row=row, column=0, columnspan=2, padx=PADDING, pady=(PADDING, PADDING_XS), sticky="w")
        row += 1

        settings = [
            ("Sensitivity阈值:", 0.95, "sensitivity_threshold"),
            ("AGBD最小值(Mg/ha):", 0, "agbd_min"),
            ("AGBD最大值(Mg/ha):", 500, "agbd_max"),
        ]
        for label_text, default, attr_name in settings:
            self._make_label(f, label_text, width=160).grid(row=row, column=0, padx=(PADDING, PADDING_SM), pady=PADDING_XS, sticky="w")
            var = ctk.DoubleVar(value=default)
            setattr(self, attr_name, var)
            self._make_entry(f, textvariable=var, width=120).grid(
                row=row, column=1, padx=(PADDING_SM, PADDING), pady=PADDING_XS, sticky="w")
            row += 1

        self.enable_time_filter = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            f, text="启用时间筛选", variable=self.enable_time_filter, font=FONT_BODY,
            text_color=FG_PRIMARY, checkbox_width=20, checkbox_height=20,
        ).grid(row=row, column=0, columnspan=2, padx=PADDING, pady=PADDING_XS, sticky="w")
        row += 1

        time_settings = [
            ("参考日期:", "2022-11-22", "reference_date", True),
            ("时间窗口(天):", 16, "time_window", False),
        ]
        for label_text, default, attr_name, is_str in time_settings:
            self._make_label(f, label_text, width=160).grid(row=row, column=0, padx=(PADDING, PADDING_SM), pady=PADDING_XS, sticky="w")
            if is_str:
                var = ctk.StringVar(value=default)
            else:
                var = ctk.IntVar(value=default)
            setattr(self, attr_name, var)
            self._make_entry(f, textvariable=var, width=120).grid(
                row=row, column=1, padx=(PADDING_SM, PADDING), pady=PADDING_XS, sticky="w")
            row += 1

    def _toggle_advanced(self):
        if self.advanced_visible:
            self.advanced_frame.grid_forget()
            self.advanced_toggle.configure(text="▶ 高级设置")
            self.advanced_visible = False
        else:
            self.advanced_frame.grid(row=5, column=0, columnspan=2, padx=PADDING, pady=PADDING_XS, sticky="ew")
            self.advanced_toggle.configure(text="▼ 高级设置")
            self.advanced_visible = True

    # ==================== 文件浏览 ====================

    def _browse_aoi_download(self):
        fp = filedialog.askopenfilename(title="选择AOI文件", filetypes=[("GeoJSON", "*.geojson *.json"), ("All", "*.*")])
        if fp: self.aoi_file_download.set(fp)

    def _browse_download_dir(self):
        dp = filedialog.askdirectory(title="选择下载目录")
        if dp: self.download_dir.set(dp)

    def _browse_data_dir(self):
        dp = filedialog.askdirectory(title="选择H5数据目录")
        if dp: self.data_dir.set(dp)

    def _browse_aoi_process(self):
        fp = filedialog.askopenfilename(title="选择AOI文件", filetypes=[("GeoJSON", "*.geojson *.json"), ("All", "*.*")])
        if fp: self.aoi_file_process.set(fp)

    # ==================== 搜索 ====================

    def _search_data(self):
        if self.is_downloading: return
        if not self._validate_download_inputs(): return
        if self.save_credentials_var.get():
            self.config["earthdata_username"] = self.username.get().strip()
            self.config["earthdata_password"] = self.password.get().strip()
            _save_credentials(self.config)
        self.is_downloading = True
        self.search_btn.configure(state="disabled")
        self.download_btn.configure(state="disabled")
        self.progress_download.set(0.2)
        self.status_download.configure(text="正在搜索...", text_color=INFO)
        threading.Thread(target=self._search_thread, daemon=True).start()

    def _validate_download_inputs(self):
        if not self.username.get().strip():
            messagebox.showerror("错误", "请输入EarthData用户名"); return False
        if not self.password.get().strip():
            messagebox.showerror("错误", "请输入EarthData密码"); return False
        if not self.aoi_file_download.get().strip() or not os.path.exists(self.aoi_file_download.get()):
            messagebox.showerror("错误", "请选择有效的AOI文件"); return False
        return True

    def _search_thread(self):
        try:
            import earthaccess
            import geopandas as gpd
            from shapely.geometry.polygon import orient

            os.environ["EARTHDATA_USERNAME"] = self.username.get().strip()
            os.environ["EARTHDATA_PASSWORD"] = self.password.get().strip()

            self.app.after(0, lambda: self.progress_download.set(0.3))
            earthaccess.login()

            self.app.after(0, lambda: self.progress_download.set(0.5))
            gdf = gpd.read_file(self.aoi_file_download.get())
            aoi_wgs84 = gdf.to_crs("EPSG:4326")
            aoi_geometry = aoi_wgs84.geometry.iloc[0]

            if hasattr(aoi_geometry, "geoms"):
                polygon = max(aoi_geometry.geoms, key=lambda x: x.area if hasattr(x, "area") else 0)
            else:
                polygon = aoi_geometry

            oriented_polygon = orient(polygon, sign=1.0)
            coords = list(oriented_polygon.exterior.coords)
            polygon_coords = [(float(c[0]), float(c[1])) for c in coords]
            if polygon_coords[0] != polygon_coords[-1]:
                polygon_coords.append(polygon_coords[0])

            self.app.after(0, lambda: self.progress_download.set(0.7))
            results = earthaccess.search_data(
                short_name="GEDI_L4A_AGB_Density_V2_1_2056",
                polygon=polygon_coords,
                temporal=(self.start_date.get().strip(), self.end_date.get().strip()),
            )
            self.search_results = results

            for widget in self.results_frame.winfo_children():
                widget.destroy()

            if len(results) == 0:
                ctk.CTkLabel(self.results_frame, text="未找到符合条件的数据", font=FONT_SMALL, text_color=WARNING).pack(pady=PADDING_SM)
            else:
                for i, result in enumerate(results[:20]):
                    name = result["meta"]["native-id"] if "meta" in result and "native-id" in result["meta"] else f"数据集 {i + 1}"
                    ctk.CTkLabel(self.results_frame, text=f"  {i + 1}. {name}", font=FONT_SMALL, anchor="w", text_color="#e2e8f0").pack(fill="x", padx=PADDING_XS, pady=2)

            self.app.after(0, lambda: self.progress_download.set(1.0))
            self.app.after(0, lambda: self.status_download.configure(text=f"找到 {len(results)} 个数据集", text_color=ACCENT))
            self.app.after(0, lambda: self.download_btn.configure(state="normal"))

        except Exception as e:
            self.app.after(0, lambda: self.progress_download.set(0))
            self.app.after(0, lambda: self.status_download.configure(text=f"错误: {str(e)[:50]}", text_color=DANGER))
            self.app.after(0, lambda: messagebox.showerror("错误", f"搜索失败:\n{str(e)}"))
        finally:
            self.is_downloading = False
            self.app.after(0, lambda: self.search_btn.configure(state="normal"))

    # ==================== 下载 ====================

    def _download_data(self):
        if self.is_downloading: return
        if self.save_credentials_var.get():
            self.config["earthdata_username"] = self.username.get().strip()
            self.config["earthdata_password"] = self.password.get().strip()
            _save_credentials(self.config)
        if self.download_method.get() == "harmony":
            self._start_harmony_download()
        else:
            self._start_earthaccess_download()

    def _start_earthaccess_download(self):
        if len(self.search_results) == 0:
            messagebox.showerror("错误", "没有可下载的数据，请先搜索"); return
        download_path = self.download_dir.get().strip() or "./data"
        os.makedirs(download_path, exist_ok=True)
        self.is_downloading = True
        self.download_btn.configure(state="disabled")
        self.progress_download.set(0)
        threading.Thread(target=self._download_earthaccess, args=(download_path,), daemon=True).start()

    def _download_earthaccess(self, download_path):
        try:
            import earthaccess
            total = len(self.search_results)
            for i, result in enumerate(self.search_results):
                progress = (i + 1) / total * 0.9 + 0.1
                self.app.after(0, lambda p=progress: self.progress_download.set(p))
                self.app.after(0, lambda i=i: self.status_download.configure(text=f"下载 {i + 1}/{total}..."))
                earthaccess.download([result], local_path=download_path)

            self.app.after(0, lambda: self.progress_download.set(1.0))
            self.app.after(0, lambda: self.status_download.configure(text=f"下载完成! 共 {total} 个文件", text_color=ACCENT))
            self.app.after(0, lambda: messagebox.showinfo("完成", f"下载完成!\n共 {total} 个文件\n保存至: {download_path}"))
        except Exception as e:
            self.app.after(0, lambda: self.progress_download.set(0))
            self.app.after(0, lambda: self.status_download.configure(text=f"错误: {str(e)[:50]}", text_color=DANGER))
            self.app.after(0, lambda: messagebox.showerror("错误", f"下载失败:\n{str(e)}"))
        finally:
            self.is_downloading = False
            self.app.after(0, lambda: self.download_btn.configure(state="normal"))

    def _start_harmony_download(self):
        aoi_file = self.aoi_file_download.get().strip()
        if not aoi_file or not os.path.exists(aoi_file):
            messagebox.showerror("错误", "请选择有效的AOI文件"); return
        download_path = self.download_dir.get().strip() or "./data"
        os.makedirs(download_path, exist_ok=True)
        self.is_downloading = True
        self.download_btn.configure(state="disabled")
        self.progress_download.set(0.05)
        threading.Thread(
            target=self._download_harmony,
            args=(aoi_file, self.start_date.get().strip(), self.end_date.get().strip(), download_path),
            daemon=True,
        ).start()

    def _download_harmony(self, aoi_file, start_date, end_date, output_dir):
        import time
        from datetime import datetime
        import geopandas as gpd
        from shapely.geometry.polygon import orient

        try:
            if not HARMONY_AVAILABLE:
                raise Exception("Harmony 库未安装。请运行: pip install harmony-py")

            username = self.username.get().strip()
            password = self.password.get().strip()
            if not username or not password:
                raise Exception("请输入 EarthData 用户名和密码")

            client = Client(auth=(username, password))
            self.app.after(0, lambda: self.progress_download.set(0.1))

            gdf = gpd.read_file(aoi_file)
            if gdf.crs and str(gdf.crs) != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            geometry = gdf.geometry.iloc[0]
            if hasattr(geometry, "geoms"):
                geometry = max(geometry.geoms, key=lambda x: x.area if hasattr(x, "area") else 0)

            self.app.after(0, lambda: self.progress_download.set(0.15))

            collection = Collection(id=GEDI_L4A_COLLECTION_ID)
            temporal = {"start": datetime.strptime(start_date, "%Y-%m-%d"), "stop": datetime.strptime(end_date, "%Y-%m-%d")}
            bounds = gdf.total_bounds
            spatial = BBox(bounds[0], bounds[1], bounds[2], bounds[3])
            request = Request(collection=collection, spatial=spatial, temporal=temporal)

            self.app.after(0, lambda: self.progress_download.set(0.2))
            job_id = client.submit(request)
            self.app.after(0, lambda: self.status_download.configure(text=f"任务已提交 ({job_id[:8]}...)", text_color=INFO))

            state = "running"
            last_status = 0
            while state not in ["successful", "failed", "canceled", "paused"]:
                try:
                    status, state, message = client.progress(job_id)
                    if status != last_status:
                        last_status = status
                        p = 0.2 + (status / 100) * 0.6
                        self.app.after(0, lambda p=p: self.progress_download.set(p))
                    time.sleep(5)
                except Exception:
                    time.sleep(5)

            if state != "successful":
                raise Exception(f"Harmony 处理失败: {state}")

            self.app.after(0, lambda: self.progress_download.set(0.85))
            result_urls = list(client.result_urls(job_id))
            total_files = len(result_urls)
            if total_files == 0:
                raise Exception("没有找到结果文件")

            downloaded = 0
            for future in client.download_all(job_id, directory=output_dir):
                try:
                    future.result(timeout=300)
                    downloaded += 1
                    p = 0.85 + (downloaded / total_files) * 0.15
                    self.app.after(0, lambda p=p: self.progress_download.set(p))
                except Exception:
                    continue

            self.app.after(0, lambda: self.progress_download.set(1.0))
            self.app.after(0, lambda: self.status_download.configure(text=f"下载完成! 共 {downloaded} 个文件", text_color=ACCENT))
            self.app.after(0, lambda: messagebox.showinfo("完成", f"Harmony 下载完成!\n共 {downloaded} 个文件\n保存至: {output_dir}"))

        except Exception as e:
            self.app.after(0, lambda: self.progress_download.set(0))
            self.app.after(0, lambda: self.status_download.configure(text=f"错误: {str(e)[:50]}", text_color=DANGER))
            self.app.after(0, lambda: messagebox.showerror("错误", f"Harmony 下载失败:\n{str(e)}"))
        finally:
            self.is_downloading = False
            self.app.after(0, lambda: self.download_btn.configure(state="normal"))

    # ==================== 处理 ====================

    def _run_process(self):
        if self.is_processing: return
        data_dir = self.data_dir.get().strip()
        aoi_file = self.aoi_file_process.get().strip()
        output_file = self.output_file.get().strip()

        if not data_dir or not os.path.isdir(data_dir):
            messagebox.showerror("错误", "请选择有效的H5数据目录"); return
        if not aoi_file or not os.path.exists(aoi_file):
            messagebox.showerror("错误", "请选择有效的AOI文件"); return
        if not output_file:
            messagebox.showerror("错误", "请输入输出文件名"); return

        self.is_processing = True
        self.process_btn.configure(state="disabled", text="处理中...")
        self.progress_process.set(0)
        threading.Thread(target=self._process_thread, daemon=True).start()

    def _process_thread(self):
        import h5py
        import geopandas as gpd
        from shapely.geometry import Point
        from datetime import datetime, timedelta

        try:
            data_dir = self.data_dir.get().strip()
            aoi_file = self.aoi_file_process.get().strip()
            output_file = self.output_file.get().strip()
            if not output_file.lower().endswith(".gpkg"):
                output_file += ".gpkg"

            h5_files = [f for f in os.listdir(data_dir) if f.endswith(".h5")]
            if not h5_files:
                self.app.after(0, lambda: messagebox.showerror("错误", "未找到H5文件"))
                return

            all_dfs = []
            stats = {"total": 0, "after_quality": 0, "after_outlier": 0, "after_clip": 0, "corrupted": []}

            for i, h5_file in enumerate(h5_files):
                filepath = os.path.join(data_dir, h5_file)
                try:
                    with h5py.File(filepath, "r") as f:
                        if len(f.keys()) == 0:
                            stats["corrupted"].append(h5_file); continue
                        if not any(b in f for b in BEAMS):
                            stats["corrupted"].append(h5_file); continue
                except Exception:
                    stats["corrupted"].append(h5_file); continue

                progress = 0.05 + (i / len(h5_files)) * 0.5
                self.app.after(0, lambda p=progress: self.progress_process.set(p))
                self.app.after(0, lambda i=i: self.status_process.configure(text=f"处理 {i + 1}/{len(h5_files)}: {h5_file}"))

                all_data = []
                with h5py.File(filepath, "r") as f:
                    for beam in BEAMS:
                        if beam not in f: continue
                        beam_group = f[beam]
                        data = {}
                        for h5_name, var_name in VARIABLES.items():
                            if h5_name in beam_group:
                                val = beam_group[h5_name][:]
                                data[var_name] = val.astype(np.float32) if val.dtype == np.float64 else val
                        if data and "lon" in data:
                            data["beam"] = beam
                            all_data.append(pd.DataFrame(data))

                if not all_data: continue
                df = pd.concat(all_data, ignore_index=True)
                stats["total"] += len(df)

                mask = (
                    (df["l2_quality_flag"] == 1) & (df["l4_quality_flag"] == 1)
                    & ((df["degrade_flag"] == 0) | (df["degrade_flag"] == 80))
                    & (df["sensitivity"] > self.sensitivity_threshold.get())
                )
                df = df[mask].copy()
                stats["after_quality"] += len(df)

                df = df[(df["agbd"] > self.agbd_min.get()) & (df["agbd"] < self.agbd_max.get())].copy()
                stats["after_outlier"] += len(df)

                if self.enable_time_filter.get():
                    anchor = datetime(2018, 1, 1)
                    df["date"] = df["delta_time"].apply(lambda x: anchor + timedelta(seconds=float(x)))
                    try:
                        image_date = datetime.strptime(self.reference_date.get().strip(), "%Y-%m-%d")
                    except Exception:
                        image_date = datetime(2022, 11, 22)
                    window = self.time_window.get()
                    df = df[(df["date"] >= image_date - timedelta(days=window)) & (df["date"] <= image_date + timedelta(days=window))].copy()

                if len(df) > 0:
                    all_dfs.append(df)

            self.app.after(0, lambda: self.progress_process.set(0.6))

            if not all_dfs:
                self.app.after(0, lambda: messagebox.showerror("错误", "没有有效的数据"))
                return

            combined = pd.concat(all_dfs, ignore_index=True)
            anchor = datetime(2018, 1, 1)
            combined["date"] = combined["delta_time"].apply(lambda x: (anchor + timedelta(seconds=float(x))).strftime("%Y-%m-%d"))

            self.app.after(0, lambda: self.progress_process.set(0.7))
            geometry = [Point(xy) for xy in zip(combined["lon"], combined["lat"])]
            gdf = gpd.GeoDataFrame(combined, geometry=geometry, crs="EPSG:4326")

            aoi = gpd.read_file(aoi_file).to_crs("EPSG:4326")
            gdf_clipped = gpd.clip(gdf, aoi)
            stats["after_clip"] = len(gdf_clipped)

            if len(gdf_clipped) == 0:
                self.app.after(0, lambda: messagebox.showerror("错误", "研究区域内没有数据点"))
                return

            self.app.after(0, lambda: self.progress_process.set(0.85))
            gdf_final = gdf_clipped.to_crs("EPSG:32647")
            cols_to_keep = ["lon", "lat", "agbd", "agbd_se", "elev", "sensitivity", "date", "beam", "shot_number", "geometry"]
            cols_exist = [c for c in cols_to_keep if c in gdf_final.columns]
            gdf_final = gdf_final[cols_exist]
            gdf_final.to_file(output_file, driver="GPKG")

            self.app.after(0, lambda: self.progress_process.set(1.0))
            self.app.after(0, lambda: self.status_process.configure(text=f"完成! 输出: {output_file}", text_color=ACCENT))
            self.app.after(0, lambda: messagebox.showinfo("处理完成",
                f"处理完成!\n\nH5文件: {len(h5_files)}\n读取点数: {stats['total']:,}\n"
                f"质量过滤后: {stats['after_quality']:,}\n研究区域内: {stats['after_clip']:,}\n\n"
                f"AGBD: {gdf_final['agbd'].min():.1f} ~ {gdf_final['agbd'].max():.1f} Mg/ha\n"
                f"输出: {output_file}"))

        except Exception as e:
            self.app.after(0, lambda: self.progress_process.set(0))
            self.app.after(0, lambda: self.status_process.configure(text=f"错误: {str(e)[:50]}", text_color=DANGER))
            self.app.after(0, lambda: messagebox.showerror("错误", f"处理失败:\n{str(e)}"))
        finally:
            self.is_processing = False
            self.app.after(0, lambda: self.process_btn.configure(state="normal", text="开始处理"))


if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    root = ctk.CTk()
    root.title("GEDI L4A 数据下载与处理工具")
    root.geometry("650x700")
    tab = GEDITab(root, root)
    root.mainloop()
