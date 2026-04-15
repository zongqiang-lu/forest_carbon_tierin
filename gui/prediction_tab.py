"""
碳汇预测Tab - 核心预测与可视化
================================
从旧gui/main.py重构，改为CTkFrame可嵌入组件
"""

import json
import threading
import numpy as np
from pathlib import Path

# 确保HierarchicalClassifier可以被pickle反序列化
# 模型保存时HierarchicalClassifier的__module__是__main__，需要注入到当前模块
import sys
import importlib.util
_hc_spec = importlib.util.spec_from_file_location(
    "hc_module",
    str(Path(__file__).resolve().parent.parent / "src" / "06b_hierarchical_classifier.py")
)
_hc_mod = importlib.util.module_from_spec(_hc_spec)
_hc_spec.loader.exec_module(_hc_mod)
import __main__
__main__.HierarchicalClassifier = _hc_mod.HierarchicalClassifier

import customtkinter as ctk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from matplotlib.patches import Patch as MplPatch

# 设置中文字体 - 尝试多种方式确保中文正常显示
def _setup_matplotlib_chinese():
    """配置matplotlib中文字体"""
    candidates = ["Microsoft YaHei", "SimHei", "STSong", "Arial Unicode MS", "WenQuanYi Micro Hei"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    # 回退: 使用系统默认
    plt.rcParams["font.sans-serif"] = ["sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    return None

_MPL_CN_FONT = _setup_matplotlib_chinese()

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

try:
    import fiona
    HAS_FIONA = True
except ImportError:
    HAS_FIONA = False

from gui.styles import (
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BG_INPUT, BG_CODE,
    FG_PRIMARY, FG_SECONDARY, FG_MUTED,
    ACCENT, ACCENT_HOVER, ACCENT_LIGHT, DANGER,
    MPL_FIGURE_BG, MPL_AXES_BG, MPL_TEXT, MPL_SPINE,
    FONT_SUBTITLE, FONT_BODY, FONT_SMALL, FONT_MONO,
    PADDING, PADDING_SM, PADDING_XS, CARD_RADIUS, PROJECT_ROOT,
    CLASS_COLORS_RGBA, CLASS_LABELS, BORDER, PROGRESS_HEIGHT,
)

# matplotlib分类颜色映射 - 0值设为透明，矢量范围外不显示
CLASS_CMAP = mcolors.ListedColormap(
    [(0, 0, 0, 0), (239/255, 68/255, 68/255, 1), (245/255, 158/255, 11/255, 1), (22/255, 163/255, 74/255, 1)]
)
CLASS_BOUNDS = [0, 1, 2, 3, 4]
CLASS_NORM = mcolors.BoundaryNorm(CLASS_BOUNDS, CLASS_CMAP.N)


class ChineseNavigationToolbar(NavigationToolbar2Tk):
    """汉化版matplotlib导航工具栏"""

    # 按钮文字汉化映射
    _CN_TEXT = {
        "Home": "首页",
        "Back": "后退",
        "Forward": "前进",
        "Pan": "平移",
        "Zoom": "缩放",
        "Subplots": "配置",
        "Save": "保存",
        "Customize": "自定义",
    }

    def __init__(self, canvas, window, **kwargs):
        """初始化汉化工具栏"""
        super().__init__(canvas, window, **kwargs)
        self._localize_buttons()

    def _localize_buttons(self):
        """将工具栏按钮文字汉化，保留原有按钮引用（不销毁重建）"""
        from tkinter import Button as TkButton
        cn_font = ("Microsoft YaHei UI", 9)
        for name, btn in self._buttons.items():
            if not isinstance(btn, TkButton):
                continue
            cn_text = self._CN_TEXT.get(name, btn.cget("text"))
            btn.configure(
                text=cn_text,
                font=cn_font,
                bg="#f1f5f9", fg="#0f172a",
                activebackground="#e2e8f0", activeforeground="#0f172a",
                relief="flat", bd=1,
                padx=6, pady=2,
                cursor="hand2",
            )


class PredictionTab:
    """碳汇预测Tab页"""

    def __init__(self, parent: ctk.CTkFrame, app: ctk.CTk):
        self.parent = parent
        self.app = app

        # 状态
        self.model = None
        self.model_region = None
        self.config = None
        self.selected_features = []
        self.all_feature_names = []
        self.result_data = None
        self.result_meta = None
        self.predicting = False
        self._initialized = False  # 延迟加载标记
        self._boundary_geom = None  # 缓存矢量边界
        self._boundary_region = None  # 缓存边界对应的研究区
        self._boundary_crs = None  # 缓存边界的坐标系

        # 延迟导入
        try:
            import rasterio
            self._rasterio = rasterio
            HAS_RASTERIO = True
        except ImportError:
            HAS_RASTERIO = False
        self.HAS_RASTERIO = HAS_RASTERIO

        try:
            import joblib
            self._joblib = joblib
            HAS_JOBLIB = True
        except ImportError:
            HAS_JOBLIB = False
        self.HAS_JOBLIB = HAS_JOBLIB

        try:
            from PIL import Image
            self._PIL_Image = Image
            HAS_PIL = True
        except ImportError:
            HAS_PIL = False
        self.HAS_PIL = HAS_PIL

        self._build_ui()
        # 延迟加载：不在初始化时加载模型，等首次切换到此Tab时再加载

    def _make_card(self, parent, **kw):
        """创建统一样式卡片"""
        return ctk.CTkFrame(
            parent, fg_color=BG_SECONDARY, corner_radius=CARD_RADIUS,
            border_width=1, border_color=BORDER, **kw
        )

    def _build_ui(self):
        """构建界面"""
        self.main_frame = ctk.CTkFrame(self.parent, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True)

        self.main_frame.grid_columnconfigure(0, weight=3)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # ===== 顶部工具栏 =====
        toolbar = self._make_card(self.main_frame)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=PADDING_SM, pady=(PADDING_SM, PADDING_XS))
        toolbar.grid_columnconfigure(5, weight=1)

        # 研究区选择
        ctk.CTkLabel(toolbar, text="研究区:", font=FONT_BODY, text_color=FG_SECONDARY).grid(
            row=0, column=0, padx=(PADDING, PADDING_XS), pady=PADDING_SM)
        self.region_var = ctk.StringVar(value="ninger")
        region_menu = ctk.CTkOptionMenu(
            toolbar, values=["ninger", "shuangbai"], variable=self.region_var,
            width=130, height=34, font=FONT_BODY,
            fg_color=BG_INPUT, button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            text_color=FG_PRIMARY,
        )
        region_menu.grid(row=0, column=1, padx=PADDING_XS, pady=PADDING_SM)
        region_menu.configure(command=self._on_region_change)

        # 年份选择
        ctk.CTkLabel(toolbar, text="年份:", font=FONT_BODY, text_color=FG_SECONDARY).grid(
            row=0, column=2, padx=PADDING_XS, pady=PADDING_SM)
        self.year_var = ctk.IntVar(value=2019)
        ctk.CTkOptionMenu(
            toolbar, values=["2019", "2023", "2024", "2025"], variable=self.year_var,
            width=100, height=34, font=FONT_BODY,
            fg_color=BG_INPUT, button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            text_color=FG_PRIMARY,
        ).grid(row=0, column=3, padx=PADDING_XS, pady=PADDING_SM)

        # 操作按钮
        ctk.CTkButton(
            toolbar, text="加载模型", width=100, height=34, font=FONT_BODY,
            fg_color=BG_TERTIARY, hover_color=BG_SECONDARY, text_color=FG_PRIMARY,
            command=self.load_model,
        ).grid(row=0, column=4, padx=PADDING_XS, pady=PADDING_SM)

        ctk.CTkButton(
            toolbar, text="一键预测", width=110, height=34, font=FONT_BODY,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#ffffff",
            command=self.run_prediction,
        ).grid(row=0, column=5, padx=PADDING_XS, pady=PADDING_SM, sticky="e")

        # 进度条
        self.progress = ctk.CTkProgressBar(toolbar, width=180, height=PROGRESS_HEIGHT,
                                           fg_color=BG_TERTIARY, progress_color=ACCENT)
        self.progress.grid(row=0, column=6, padx=(PADDING_XS, PADDING), pady=PADDING_SM, sticky="e")
        self.progress.set(0)

        # ===== 左侧: 结果展示 =====
        left_card = self._make_card(self.main_frame)
        left_card.grid(row=1, column=0, sticky="nsew", padx=(PADDING_SM, PADDING_XS), pady=PADDING_SM)

        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor=MPL_FIGURE_BG)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(MPL_AXES_BG)
        self.ax.set_title("等待预测...", color=MPL_TEXT, fontsize=14)
        self.ax.axis("off")
        for spine in self.ax.spines.values():
            spine.set_edgecolor(MPL_SPINE)
        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, left_card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        nav_frame = ctk.CTkFrame(left_card, fg_color="transparent", height=30)
        nav_frame.pack(fill="x", padx=4, pady=(0, 4))
        self.toolbar_nav = ChineseNavigationToolbar(self.canvas, nav_frame)
        self.toolbar_nav.update()

        # ===== 右侧: 信息面板 =====
        right_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent", width=320)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(PADDING_XS, PADDING_SM), pady=PADDING_SM)
        right_frame.grid_propagate(False)

        # 模型信息卡片
        model_card = self._make_card(right_frame)
        model_card.pack(fill="x", pady=(0, PADDING_SM))
        ctk.CTkLabel(
            model_card, text="模型信息", font=FONT_SUBTITLE, text_color=ACCENT,
        ).pack(anchor="w", padx=PADDING, pady=(PADDING, PADDING_XS))

        self.model_info_text = ctk.CTkTextbox(
            model_card, height=150, font=FONT_MONO,
            fg_color=BG_CODE, text_color="#e2e8f0",
            corner_radius=8, state="disabled", border_width=0,
        )
        self.model_info_text.pack(fill="x", padx=PADDING, pady=(0, PADDING))

        # 统计信息卡片
        stats_card = self._make_card(right_frame)
        stats_card.pack(fill="x", pady=(0, PADDING_SM))
        ctk.CTkLabel(
            stats_card, text="统计信息", font=FONT_SUBTITLE, text_color=ACCENT,
        ).pack(anchor="w", padx=PADDING, pady=(PADDING, PADDING_XS))

        self.stats_info_text = ctk.CTkTextbox(
            stats_card, height=160, font=FONT_MONO,
            fg_color=BG_CODE, text_color="#e2e8f0",
            corner_radius=8, state="disabled", border_width=0,
        )
        self.stats_info_text.pack(fill="x", padx=PADDING, pady=(0, PADDING))

        # 日志卡片
        log_card = self._make_card(right_frame)
        log_card.pack(fill="both", expand=True)
        ctk.CTkLabel(
            log_card, text="运行日志", font=FONT_SUBTITLE, text_color=ACCENT,
        ).pack(anchor="w", padx=PADDING, pady=(PADDING, PADDING_XS))

        self.log_text = ctk.CTkTextbox(
            log_card, font=FONT_MONO,
            fg_color=BG_CODE, text_color="#94a3b8",
            corner_radius=8, border_width=0,
        )
        self.log_text.pack(fill="both", expand=True, padx=PADDING, pady=(0, PADDING))

        # ===== 底部导出栏 =====
        export_bar = self._make_card(self.main_frame)
        export_bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=PADDING_SM, pady=(PADDING_XS, PADDING_SM))

        ctk.CTkButton(
            export_bar, text="导出 TIF", width=120, height=36, font=FONT_BODY,
            fg_color=BG_TERTIARY, hover_color=BG_SECONDARY, text_color=FG_PRIMARY,
            command=self.export_result,
        ).pack(side="left", padx=(PADDING, PADDING_XS), pady=PADDING_SM)

        ctk.CTkButton(
            export_bar, text="导出 PNG", width=120, height=36, font=FONT_BODY,
            fg_color=BG_TERTIARY, hover_color=BG_SECONDARY, text_color=FG_PRIMARY,
            command=self.export_png,
        ).pack(side="left", padx=PADDING_XS, pady=PADDING_SM)

    def log(self, msg: str):
        """输出日志"""
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")
        self.app.update_idletasks()

    def _on_region_change(self, value):
        self._load_default_model()

    def ensure_initialized(self):
        """延迟初始化 - 首次切换到此Tab时调用"""
        if not self._initialized:
            self._initialized = True
            self._load_default_model()

    def _load_default_model(self):
        """加载默认模型（从model_metrics.json读取模型类型，加载rf_model.pkl）"""
        region = self.region_var.get()

        # 模型文件统一为rf_model.pkl（06b脚本也将HierarchicalClassifier保存到此路径）
        model_path = PROJECT_ROOT / f"output/{region}/2019/metrics/rf_model.pkl"
        # 也兼容旧版hc_model.pkl
        hc_model_path = PROJECT_ROOT / f"output/{region}/2019/metrics/hc_model.pkl"

        metrics_path = PROJECT_ROOT / f"output/{region}/2019/metrics/model_metrics.json"
        fnames_path = PROJECT_ROOT / f"data/{region}/2019/aligned/feature_names.json"

        # 从metrics JSON中读取真实模型类型
        model_type_label = "Unknown"
        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics_info = json.load(f)
                model_type_label = metrics_info.get("model_type", "Unknown")
            except Exception:
                pass

        # 选择模型文件路径
        actual_model_path = None
        if model_path.exists() and self.HAS_JOBLIB:
            actual_model_path = model_path
        elif hc_model_path.exists() and self.HAS_JOBLIB:
            actual_model_path = hc_model_path

        if actual_model_path and self.HAS_JOBLIB:
            try:
                self.model = self._joblib.load(actual_model_path)
                self.model_region = region
                self.log(f"已加载模型: {actual_model_path.name} ({model_type_label})")

                if fnames_path.exists():
                    with open(fnames_path, "r", encoding="utf-8") as f:
                        self.all_feature_names = json.load(f)["feature_names"]

                import yaml
                cfg_path = PROJECT_ROOT / f"config/{region}.yaml"
                if cfg_path.exists():
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)
                    self.selected_features = cfg.get("selected_features", self.all_feature_names)
                    # 空列表表示使用全部特征
                    if not self.selected_features:
                        self.selected_features = self.all_feature_names

                if metrics_path.exists():
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                    region_name = "宁洱县" if region == "ninger" else "双柏县"
                    cv_line = ""
                    if "cv_oa_mean" in metrics and "cv_oa_std" in metrics:
                        cv_line = f"  CV OA:    {metrics['cv_oa_mean']:.3f} ± {metrics['cv_oa_std']:.3f}\n"
                    info = (
                        f"  模型类型: {metrics['model_type']}\n"
                        f"  研究区:   {region_name} ({region})\n"
                        f"  特征数:   {metrics['n_features']}\n"
                        + cv_line +
                        f"  Val OA:   {metrics['val_oa']:.3f}\n"
                        f"  Kappa:    {metrics['val_kappa']:.3f}\n"
                        f"  训练样本: {metrics['n_train']:,}\n"
                        f"  验证样本: {metrics['n_val']:,}"
                    )
                    self.model_info_text.configure(state="normal")
                    self.model_info_text.delete("1.0", "end")
                    self.model_info_text.insert("end", info)
                    self.model_info_text.configure(state="disabled")

                self.app.set_status(f"模型已加载: {region}/2019 ({model_type_label}) | 点击一键预测开始")
            except Exception as e:
                self.log(f"加载模型失败: {e}")
        else:
            self.log("未找到默认模型，请手动加载")

    def load_model(self):
        """手动加载模型"""
        path = filedialog.askopenfilename(
            title="选择RF模型文件",
            filetypes=[("Pickle文件", "*.pkl"), ("所有文件", "*.*")],
            initialdir=str(PROJECT_ROOT / "output"),
        )
        if path and self.HAS_JOBLIB:
            try:
                self.model = self._joblib.load(path)
                self.log(f"已加载模型: {Path(path).name}")
                self.app.set_status(f"模型已加载: {Path(path).name}")
            except Exception as e:
                messagebox.showerror("错误", f"加载模型失败: {e}")

    def run_prediction(self):
        """一键预测（异步）"""
        if self.predicting:
            messagebox.showinfo("提示", "正在预测中，请等待完成")
            return
        if not self.model:
            messagebox.showwarning("警告", "请先加载模型")
            return
        if not self.HAS_RASTERIO:
            messagebox.showerror("错误", "需要rasterio库")
            return

        self.predicting = True
        self.progress.set(0.1)
        self.app.set_status("正在预测...")

        thread = threading.Thread(target=self._do_prediction, daemon=True)
        thread.start()

    def _do_prediction(self):
        """实际预测逻辑（后台线程）"""
        region = self.region_var.get()
        year = self.year_var.get()

        feature_path = PROJECT_ROOT / f"data/{region}/{year}/aligned/feature_stack.tif"
        if not feature_path.exists():
            self.app.after(0, lambda: messagebox.showerror("错误", f"特征栈不存在: {feature_path}"))
            self._finish_prediction()
            return

        # 加载目标年份的feature_names（可能与训练年不同）
        year_feature_names_path = PROJECT_ROOT / f"data/{region}/{year}/aligned/feature_names.json"
        if year_feature_names_path.exists():
            try:
                with open(year_feature_names_path, "r", encoding="utf-8") as f:
                    year_feature_names = json.load(f)["feature_names"]
            except Exception:
                year_feature_names = self.all_feature_names
        else:
            year_feature_names = self.all_feature_names

        self.app.after(0, lambda: self.log(f"开始预测 {region}/{year}..."))

        try:
            rasterio = self._rasterio
            with rasterio.open(feature_path) as src:
                features = src.read()
                self.result_meta = src.profile.copy()
                n_bands, height, width = features.shape

            self.app.after(0, lambda: self.log(f"影像尺寸: {width}x{height}, {n_bands}波段"))
            self.app.after(0, lambda: self.progress.set(0.2))

            mask_path = PROJECT_ROOT / f"data/{region}/{year}/aligned/forest_mask.tif"
            if mask_path.exists():
                with rasterio.open(mask_path) as src:
                    forest = src.read(1)
                valid = forest == 1
            else:
                valid = features[0] != 0

            if self.selected_features and year_feature_names:
                selected_idx = [year_feature_names.index(n) for n in self.selected_features if n in year_feature_names]
            else:
                selected_idx = list(range(n_bands))

            features_flat = features.reshape(n_bands, -1).T
            valid_flat = valid.flatten()
            features_valid = features_flat[valid_flat]
            features_selected = features_valid[:, selected_idx]

            # 验证特征维度与模型匹配
            n_selected = len(selected_idx)
            if hasattr(self.model, 'n_features_in_') and n_selected != self.model.n_features_in_:
                self.app.after(0, lambda: self.log(
                    f"特征维度不匹配: 特征栈{n_selected}维 vs 模型期望{self.model.n_features_in_}维。"
                    f"请重新生成该年份的特征栈。"
                ))
                self.app.after(0, lambda: messagebox.showerror(
                    "错误",
                    f"特征维度不匹配: 特征栈{n_selected}维 vs 模型期望{self.model.n_features_in_}维\n"
                    f"请用 --force 重新运行 pipeline: python src/run_all.py --region {region} --year {year} --force"
                ))
                self._finish_prediction()
                return

            # 允许最多30%的NaN特征（与训练时一致）
            nan_ratio = np.isnan(features_selected).sum(axis=1) / features_selected.shape[1]
            valid_features_mask = nan_ratio < 0.3

            # 仅保留通过NaN过滤的像素进行预测
            features_selected = features_selected[valid_features_mask]

            # 对NaN特征用列均值填充
            if np.isnan(features_selected).any():
                col_means = np.nanmean(features_selected, axis=0)
                nan_mask = np.isnan(features_selected)
                for j in range(features_selected.shape[1]):
                    features_selected[nan_mask[:, j], j] = col_means[j] if np.isfinite(col_means[j]) else 0

            n_predict = len(features_selected)
            self.app.after(0, lambda: self.log(f"预测特征: {len(selected_idx)}维, {n_predict}像素 (过滤NaN后)"))
            self.app.after(0, lambda: self.progress.set(0.4))

            batch_size = 100000
            predictions = []
            total_batches = (n_predict + batch_size - 1) // batch_size
            for i in range(0, n_predict, batch_size):
                batch = features_selected[i:i + batch_size]
                pred = self.model.predict(batch)
                predictions.append(pred)
                batch_num = i // batch_size + 1
                prog = 0.4 + 0.5 * (batch_num / total_batches)
                self.app.after(0, lambda bn=batch_num, p=prog: (
                    self.log(f"  批次 {bn}: {len(batch)} 像素"),
                    self.progress.set(p),
                ))

            all_pred = np.concatenate(predictions)

            # 构建结果：仅有效特征且通过NaN过滤的像元才赋值
            valid_pixel_indices = np.where(valid_flat)[0]
            result = np.zeros(height * width, dtype=np.uint8)
            result[valid_pixel_indices[valid_features_mask]] = all_pred.astype(np.uint8)
            self.result_data = result.reshape(height, width)

            self.app.after(0, lambda: self.progress.set(0.95))
            self.app.after(0, self._display_result)
            self.app.after(0, self._compute_stats)
            self.app.after(0, lambda: self.log(f"预测完成! 有效像元: {len(all_pred):,}"))
            self.app.after(0, lambda: self.app.set_status(f"预测完成: {region}/{year}"))

        except Exception as e:
            self.app.after(0, lambda: self.log(f"预测失败: {e}"))
            self.app.after(0, lambda: messagebox.showerror("错误", f"预测失败: {e}"))
            self.app.after(0, lambda: self.app.set_status("预测失败"))

        self._finish_prediction()

    def _finish_prediction(self):
        self.predicting = False
        self.progress.set(1.0 if self.result_data is not None else 0)

    def _display_result(self):
        """显示预测结果 - 仅显示矢量边界范围内数据"""
        if self.result_data is None:
            return

        self.ax.clear()

        # 应用矢量边界掩膜：将边界范围外的像元设为0（透明）
        display_data = self._apply_vector_mask(self.result_data)

        self.ax.imshow(display_data, cmap=CLASS_CMAP, norm=CLASS_NORM, interpolation="nearest")
        self.ax.set_title("碳汇分级结果", color=MPL_TEXT, fontsize=14, fontweight="bold")
        self.ax.axis("off")

        legend_elements = [
            Patch(facecolor="#ef4444", label="低碳密度 (Low)"),
            Patch(facecolor="#f59e0b", label="中碳密度 (Medium)"),
            Patch(facecolor="#16a34a", label="高碳密度 (High)"),
        ]
        self.ax.legend(
            handles=legend_elements, loc="upper right", fontsize=9,
            facecolor=MPL_FIGURE_BG, edgecolor=MPL_SPINE, labelcolor=MPL_TEXT,
        )

        # 绘制矢量边界线
        self._draw_boundary()

        for spine in self.ax.spines.values():
            spine.set_edgecolor(MPL_SPINE)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

    def _load_boundary(self, region: str):
        """加载研究区矢量边界（优先加载与栅格CRS匹配的UTM边界）"""
        # 每次都根据当前region重新加载，避免缓存错误
        # 优先从data目录加载UTM坐标系的边界（与栅格数据同坐标系）
        boundary_paths = [
            PROJECT_ROOT / f"data/{region}/raw/boundary" / f"{'宁洱县' if region == 'ninger' else '双柏县'}_32647.geojson",
            PROJECT_ROOT / f"data/{region}/raw/boundary" / f"{'宁洱县' if region == 'ninger' else '双柏县'}_4326.geojson",
            PROJECT_ROOT / f"web/backend/data/boundaries/{region}.geojson",
        ]

        import json as _json
        for bp in boundary_paths:
            if bp.exists():
                try:
                    with open(bp, "r", encoding="utf-8") as f:
                        geojson_data = _json.load(f)
                    self._boundary_geom = geojson_data
                    self._boundary_region = region
                    self._boundary_crs = "EPSG:32647" if "_32647" in bp.name else "EPSG:4326"
                    self.log(f"已加载矢量边界: {bp.name} ({self._boundary_crs})")
                    return geojson_data
                except Exception as e:
                    self.log(f"加载边界失败 {bp.name}: {e}")

        self._boundary_geom = None
        self._boundary_region = None
        self._boundary_crs = None
        return None

    def _apply_vector_mask(self, data: np.ndarray) -> np.ndarray:
        """用矢量边界掩膜预测结果，范围外设为0"""
        if not self.HAS_RASTERIO or self.result_meta is None:
            return data

        region = self.region_var.get()
        geojson_data = self._load_boundary(region)
        if geojson_data is None:
            return data

        try:
            from rasterio.features import geometry_mask
            from rasterio.warp import transform as warp_transform
            from pyproj import CRS

            geometries = []
            if geojson_data.get("type") == "FeatureCollection":
                for feat in geojson_data.get("features", []):
                    geometries.append(feat["geometry"])
            elif geojson_data.get("type") in ("Polygon", "MultiPolygon"):
                geometries.append(geojson_data)
            elif geojson_data.get("geometry"):
                geometries.append(geojson_data["geometry"])

            if not geometries:
                return data

            # 检查GeoJSON坐标与栅格CRS是否一致，如果不一致则转换坐标
            raster_crs = self.result_meta.get("crs")
            boundary_crs_str = getattr(self, '_boundary_crs', None)

            if raster_crs and boundary_crs_str == "EPSG:4326":
                # GeoJSON是WGS84，栅格是UTM，需要转换几何坐标
                try:
                    raster_epsg = raster_crs.to_epsg() if hasattr(raster_crs, 'to_epsg') else None
                    if raster_epsg and raster_epsg != 4326:
                        from shapely.geometry import shape, mapping
                        from shapely.ops import transform as shapely_transform
                        from functools import partial
                        from pyproj import Transformer

                        transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)

                        converted_geoms = []
                        for geom in geometries:
                            shapely_geom = shape(geom)
                            projected = shapely_transform(transformer.transform, shapely_geom)
                            converted_geoms.append(mapping(projected))
                        geometries = converted_geoms
                        self.log("边界坐标已从WGS84转换到栅格CRS")
                except ImportError:
                    self.log("shapely/pyproj未安装，跳过坐标转换（掩膜可能不准确）")
                except Exception as e:
                    self.log(f"边界坐标转换失败: {e}")

            mask = geometry_mask(
                geometries,
                out_shape=data.shape,
                transform=self.result_meta["transform"],
                invert=True,
            )
            result = data.copy()
            result[~mask] = 0  # 边界范围外设为0
            return result
        except Exception as e:
            self.log(f"矢量掩膜失败，使用原始数据: {e}")
            return data

    def _draw_boundary(self):
        """在图上绘制矢量边界线"""
        region = self.region_var.get()
        geojson_data = self._load_boundary(region)
        if geojson_data is None or not self.HAS_RASTERIO or self.result_meta is None:
            return

        try:
            from shapely.geometry import shape, mapping
            from shapely.ops import transform as shapely_transform
            from matplotlib.collections import LineCollection

            transform = self.result_meta["transform"]

            # 检查是否需要坐标转换
            raster_crs = self.result_meta.get("crs")
            boundary_crs_str = getattr(self, '_boundary_crs', None)
            need_transform = False
            transformer = None

            if raster_crs and boundary_crs_str == "EPSG:4326":
                try:
                    raster_epsg = raster_crs.to_epsg() if hasattr(raster_crs, 'to_epsg') else None
                    if raster_epsg and raster_epsg != 4326:
                        from pyproj import Transformer
                        transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
                        need_transform = True
                except ImportError:
                    pass

            def map_to_pixel(x, y):
                col = (x - transform.c) / transform.a
                row = (y - transform.f) / transform.e
                return col, row

            shapes = []
            if geojson_data.get("type") == "FeatureCollection":
                for feat in geojson_data.get("features", []):
                    geom = shape(feat["geometry"])
                    if need_transform and transformer:
                        geom = shapely_transform(transformer.transform, geom)
                    shapes.append(geom)
            elif geojson_data.get("type") in ("Polygon", "MultiPolygon"):
                geom = shape(geojson_data)
                if need_transform and transformer:
                    geom = shapely_transform(transformer.transform, geom)
                shapes.append(geom)
            elif geojson_data.get("geometry"):
                geom = shape(geojson_data["geometry"])
                if need_transform and transformer:
                    geom = shapely_transform(transformer.transform, geom)
                shapes.append(geom)

            for geom in shapes:
                if geom.geom_type == "Polygon":
                    coords = list(geom.exterior.coords)
                    pixel_coords = [map_to_pixel(x, y) for x, y in coords]
                    self.ax.plot(
                        [p[0] for p in pixel_coords],
                        [p[1] for p in pixel_coords],
                        color="#334155", linewidth=1.5, alpha=0.7,
                    )
                    for interior in geom.interiors:
                        icoords = list(interior.coords)
                        ipixel = [map_to_pixel(x, y) for x, y in icoords]
                        self.ax.plot(
                            [p[0] for p in ipixel],
                            [p[1] for p in ipixel],
                            color="#334155", linewidth=1.0, alpha=0.5,
                        )
                elif geom.geom_type == "MultiPolygon":
                    for poly in geom.geoms:
                        coords = list(poly.exterior.coords)
                        pixel_coords = [map_to_pixel(x, y) for x, y in coords]
                        self.ax.plot(
                            [p[0] for p in pixel_coords],
                            [p[1] for p in pixel_coords],
                            color="#334155", linewidth=1.5, alpha=0.7,
                        )
        except ImportError:
            # shapely不可用时跳过边界线绘制
            self.log("shapely未安装，跳过边界线绘制")
        except Exception as e:
            self.log(f"绘制边界线失败: {e}")

    def _compute_stats(self):
        """计算并显示统计信息"""
        if self.result_data is None:
            return

        region = self.region_var.get()
        year = self.year_var.get()
        region_name = "宁洱县" if region == "ninger" else "双柏县"

        stats_lines = [f"  区域: {region_name} ({region})", f"  年份: {year}", ""]
        total = 0
        for cls in [1, 2, 3]:
            count = int(np.sum(self.result_data == cls))
            total += count

        pixel_area_km2 = 30 * 30 / 1e6
        stats_lines.append(f"  森林面积: {total * pixel_area_km2:.1f} km²")
        stats_lines.append(f"  有效像元: {total:,}")
        stats_lines.append("")

        if total > 0:
            for cls in [1, 2, 3]:
                count = int(np.sum(self.result_data == cls))
                pct = count / total * 100
                area = count * pixel_area_km2
                label = CLASS_LABELS.get(cls, f"Level{cls}")
                stats_lines.append(f"  {label}:")
                stats_lines.append(f"    {count:,} 像素 | {pct:.1f}% | {area:.1f} km²")

        self.stats_info_text.configure(state="normal")
        self.stats_info_text.delete("1.0", "end")
        self.stats_info_text.insert("end", "\n".join(stats_lines))
        self.stats_info_text.configure(state="disabled")

    def export_result(self):
        """导出TIF"""
        if self.result_data is None:
            messagebox.showwarning("警告", "没有预测结果可导出")
            return

        region = self.region_var.get()
        year = self.year_var.get()
        path = filedialog.asksaveasfilename(
            title="导出结果为GeoTIFF",
            defaultextension=".tif",
            filetypes=[("GeoTIFF", "*.tif")],
            initialfile=f"carbon_density_class_{region}_{year}.tif",
            initialdir=str(PROJECT_ROOT / "output"),
        )
        if not path:
            return

        try:
            if self.HAS_RASTERIO and self.result_meta:
                self.result_meta.update(dtype="uint8", count=1)
                with self._rasterio.open(path, "w", **self.result_meta) as dst:
                    dst.write(self.result_data, 1)
                self.log(f"已导出TIF: {Path(path).name}")
                self.app.set_status(f"已导出: {Path(path).name}")
            else:
                messagebox.showwarning("警告", "缺少rasterio库或元数据")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {e}")

    def export_png(self):
        """导出彩色PNG"""
        if self.result_data is None:
            messagebox.showwarning("警告", "没有预测结果可导出")
            return

        region = self.region_var.get()
        year = self.year_var.get()
        path = filedialog.asksaveasfilename(
            title="导出结果为PNG",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile=f"carbon_density_class_{region}_{year}.png",
            initialdir=str(PROJECT_ROOT / "output"),
        )
        if not path:
            return

        try:
            if self.HAS_PIL:
                h, w = self.result_data.shape
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                color_map = {1: [239, 68, 68, 255], 2: [245, 158, 11, 255], 3: [22, 163, 74, 255]}
                for cls, color in color_map.items():
                    mask = self.result_data == cls
                    for i, c in enumerate(color):
                        rgba[:, :, i][mask] = c
                img = self._PIL_Image.fromarray(rgba, mode="RGBA")
                img.save(path)
                self.log(f"已导出PNG: {Path(path).name}")
                self.app.set_status(f"已导出: {Path(path).name}")
            else:
                messagebox.showwarning("警告", "缺少Pillow库")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {e}")
