"""
GeoJSON坐标系转换Tab
====================
从geojson_converter/重构为CTkFrame可嵌入组件
"""

import os
import customtkinter as ctk
from tkinter import filedialog, messagebox

from gui.styles import (
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BG_INPUT,
    FG_PRIMARY, FG_SECONDARY, FG_MUTED,
    ACCENT, ACCENT_HOVER, ACCENT_LIGHT,
    FONT_SUBTITLE, FONT_BODY, FONT_SMALL,
    PADDING, PADDING_SM, PADDING_XS, CARD_RADIUS, BORDER,
)


class GeoJSONTab:
    """GeoJSON转换Tab页"""

    def __init__(self, parent: ctk.CTkFrame, app: ctk.CTk):
        self.parent = parent
        self.app = app
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

    def _make_file_row(self, parent, label_text, row, variable=None, browse_cmd=None, entry_width=None):
        """创建统一的文件选择行"""
        self._make_label(parent, label_text, width=150).grid(
            row=row, column=0, padx=(PADDING, PADDING_SM), pady=PADDING_SM, sticky="w")
        entry = self._make_entry(parent, textvariable=variable, **({} if entry_width is None else {"width": entry_width}))
        entry.grid(row=row, column=1, padx=PADDING_SM, pady=PADDING_SM, sticky="ew" if entry_width is None else "w")
        if browse_cmd:
            ctk.CTkButton(
                parent, text="浏览", width=60, height=36, font=FONT_BODY,
                fg_color=BG_TERTIARY, hover_color=BG_SECONDARY, text_color=FG_PRIMARY,
                command=browse_cmd,
            ).grid(row=row, column=2, padx=(PADDING_XS, PADDING), pady=PADDING_SM)
        return entry

    def _build_ui(self):
        """构建界面"""
        self.main_frame = ctk.CTkFrame(self.parent, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True)

        # 标题行
        title_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        title_frame.pack(fill="x", padx=PADDING, pady=(PADDING, PADDING_SM))

        ctk.CTkLabel(
            title_frame, text="GeoJSON 坐标系转换",
            font=FONT_SUBTITLE, text_color=ACCENT,
        ).pack(side="left")
        ctk.CTkLabel(
            title_frame, text="  将 GeoJSON 文件从一种坐标系转换为另一种",
            font=FONT_SMALL, text_color=FG_MUTED,
        ).pack(side="left", padx=PADDING_SM)

        # 主表单卡片
        form_card = self._make_card(self.main_frame)
        form_card.pack(fill="x", padx=PADDING, pady=PADDING_SM)
        form_card.grid_columnconfigure(1, weight=1)

        # 输入文件
        self.input_path = ctk.StringVar()
        self._make_file_row(form_card, "输入文件:", 0, self.input_path, self._browse_input)

        # 原始坐标系
        self.source_epsg = ctk.StringVar(value="4490")
        self._make_label(form_card, "原始坐标系 (EPSG):", width=150).grid(
            row=1, column=0, padx=(PADDING, PADDING_SM), pady=PADDING_SM, sticky="w")
        self._make_entry(form_card, textvariable=self.source_epsg, width=130).grid(
            row=1, column=1, padx=PADDING_SM, pady=PADDING_SM, sticky="w")

        # 目标坐标系
        self.target_epsg = ctk.StringVar(value="32647")
        self._make_label(form_card, "目标坐标系 (EPSG):", width=150).grid(
            row=2, column=0, padx=(PADDING, PADDING_SM), pady=PADDING_SM, sticky="w")
        self._make_entry(form_card, textvariable=self.target_epsg, width=130).grid(
            row=2, column=1, padx=PADDING_SM, pady=PADDING_SM, sticky="w")

        # 输出文件
        self.output_path = ctk.StringVar()
        self._make_file_row(form_card, "输出文件:", 3, self.output_path, self._browse_output)

        # 转换按钮
        btn_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=PADDING, pady=PADDING_SM)

        ctk.CTkButton(
            btn_frame, text="开始转换", width=200, height=42, font=FONT_BODY,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#ffffff",
            command=self._convert,
        ).pack(side="left")

        # 状态标签
        self.status_label = ctk.CTkLabel(
            btn_frame, text="", font=FONT_BODY, text_color=FG_MUTED,
        )
        self.status_label.pack(side="left", padx=PADDING)

        # 使用说明卡片
        info_card = self._make_card(self.main_frame)
        info_card.pack(fill="x", padx=PADDING, pady=PADDING_SM)

        ctk.CTkLabel(
            info_card, text="使用说明",
            font=(FONT_BODY[0], FONT_BODY[1], "bold"), text_color=FG_PRIMARY,
        ).pack(anchor="w", padx=PADDING, pady=(PADDING, PADDING_XS))

        tips = [
            "选择需要转换坐标系的GeoJSON文件",
            "输入原始坐标系EPSG代码（如 CGCS2000 = 4490）",
            "输入目标坐标系EPSG代码（如 WGS84/UTM Zone 47N = 32647）",
            "点击「开始转换」即可完成坐标系变换",
        ]
        for tip in tips:
            ctk.CTkLabel(
                info_card, text=f"  ·  {tip}",
                font=FONT_SMALL, text_color=FG_MUTED, anchor="w",
            ).pack(fill="x", padx=(PADDING + 8, PADDING), pady=PADDING_XS)

        ctk.CTkFrame(info_card, fg_color="transparent", height=PADDING_SM).pack()

    def _browse_input(self):
        initial_dir = os.getcwd()
        if self.input_path.get():
            initial_dir = os.path.dirname(self.input_path.get()) or os.getcwd()
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="选择GeoJSON文件",
            filetypes=[
                ("GeoJSON文件", "*.geojson"),
                ("JSON文件", "*.json"),
                ("所有文件", "*.*"),
            ],
        )
        if file_path:
            self.input_path.set(file_path)
            self._update_output_path()

    def _browse_output(self):
        initial_dir = os.getcwd()
        if self.input_path.get():
            initial_dir = os.path.dirname(self.input_path.get())
        file_path = filedialog.asksaveasfilename(
            initialdir=initial_dir,
            title="保存GeoJSON文件",
            defaultextension=".geojson",
            filetypes=[("GeoJSON文件", "*.geojson"), ("所有文件", "*.*")],
        )
        if file_path:
            self.output_path.set(file_path)

    def _update_output_path(self):
        input_file = self.input_path.get()
        if input_file:
            dir_path = os.path.dirname(input_file)
            filename = os.path.splitext(os.path.basename(input_file))[0]
            target = self.target_epsg.get() or "32647"
            output_file = os.path.join(dir_path, f"{filename}_{target}.geojson")
            self.output_path.set(output_file)

    def _convert(self):
        input_file = self.input_path.get()
        output_file = self.output_path.get()

        if not input_file:
            messagebox.showerror("错误", "请选择输入文件")
            return
        if not output_file:
            messagebox.showerror("错误", "请指定输出文件路径")
            return

        try:
            source_epsg = int(self.source_epsg.get())
            target_epsg = int(self.target_epsg.get())
        except ValueError:
            messagebox.showerror("错误", "EPSG代码必须是数字")
            return

        try:
            self.status_label.configure(text="正在转换...", text_color=ACCENT)
            self.app.update()

            import geopandas as gpd
            gdf = gpd.read_file(input_file)
            gdf = gdf.set_crs(epsg=source_epsg, allow_override=True)
            gdf_converted = gdf.to_crs(epsg=target_epsg)
            gdf_converted.to_file(output_file, driver="GeoJSON")

            self.status_label.configure(
                text=f"转换完成！ EPSG:{source_epsg} → EPSG:{target_epsg}",
                text_color=ACCENT,
            )
            messagebox.showinfo("成功", f"转换完成！\n输出文件: {output_file}")

        except Exception as e:
            self.status_label.configure(text=f"转换失败: {str(e)}", text_color="#ef4444")
            messagebox.showerror("错误", f"转换失败:\n{str(e)}")


if __name__ == "__main__":
    import customtkinter as ctk
    ctk.set_appearance_mode("light")
    root = ctk.CTk()
    root.title("GeoJSON 坐标系转换工具")
    root.geometry("650x500")
    tab = GeoJSONTab(root, root)
    root.mainloop()
