"""
Web服务管理Tab
===============
管理FastAPI后端启停、数据预处理、状态监控
"""

import os
import sys
import threading
import subprocess
import webbrowser
from pathlib import Path

import customtkinter as ctk
from tkinter import messagebox

from gui.styles import (
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BG_INPUT, BG_CODE,
    FG_PRIMARY, FG_SECONDARY, FG_MUTED,
    ACCENT, ACCENT_HOVER, ACCENT_LIGHT, ACCENT_DIM,
    DANGER, WARNING, INFO, INFO_LIGHT,
    FONT_SUBTITLE, FONT_BODY, FONT_SMALL, FONT_MONO,
    PADDING, PADDING_SM, PADDING_XS, CARD_RADIUS, PROJECT_ROOT, BORDER,
)


class WebServiceTab:
    """Web服务管理Tab页"""

    def __init__(self, parent: ctk.CTkFrame, app: ctk.CTk):
        self.parent = parent
        self.app = app
        self.server_process = None
        self.is_running = False
        self._build_ui()

    def _make_card(self, parent, **kw):
        return ctk.CTkFrame(
            parent, fg_color=BG_SECONDARY, corner_radius=CARD_RADIUS,
            border_width=1, border_color=BORDER, **kw
        )

    def _build_ui(self):
        """构建界面"""
        self.main_frame = ctk.CTkFrame(self.parent, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(3, weight=1)

        # ===== 标题行 =====
        title_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        title_frame.grid(row=0, column=0, padx=PADDING, pady=(PADDING, PADDING_SM), sticky="ew")

        ctk.CTkLabel(
            title_frame, text="Web 服务管理",
            font=FONT_SUBTITLE, text_color=ACCENT,
        ).pack(side="left")
        ctk.CTkLabel(
            title_frame, text="  FastAPI 后端 + Vue3 前端",
            font=FONT_SMALL, text_color=FG_MUTED,
        ).pack(side="left", padx=PADDING_SM)

        # ===== 状态与控制卡片 =====
        control_card = self._make_card(self.main_frame)
        control_card.grid(row=1, column=0, padx=PADDING, pady=PADDING_XS, sticky="ew")
        control_card.grid_columnconfigure(1, weight=1)

        # 第一行: 状态 + 端口
        row_frame = ctk.CTkFrame(control_card, fg_color="transparent")
        row_frame.pack(fill="x", padx=PADDING, pady=PADDING_SM)
        row_frame.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(row_frame, text="服务状态:", font=FONT_BODY, text_color=FG_SECONDARY).grid(
            row=0, column=0, padx=(0, PADDING_SM))
        self.status_indicator = ctk.CTkLabel(
            row_frame, text="● 已停止", font=FONT_BODY, text_color=DANGER, width=100, anchor="w",
        )
        self.status_indicator.grid(row=0, column=1, padx=(0, PADDING))

        ctk.CTkLabel(row_frame, text="端口:", font=FONT_BODY, text_color=FG_SECONDARY).grid(
            row=0, column=2, padx=(0, PADDING_XS), sticky="e")
        self.port_var = ctk.StringVar(value="8000")
        ctk.CTkEntry(
            row_frame, textvariable=self.port_var, font=FONT_BODY, width=80, height=36,
            fg_color=BG_INPUT, text_color=FG_PRIMARY, border_color=BORDER, border_width=1,
        ).grid(row=0, column=3, padx=(0, 0))

        # 第二行: 操作按钮
        btn_frame = ctk.CTkFrame(control_card, fg_color="transparent")
        btn_frame.pack(fill="x", padx=PADDING, pady=(0, PADDING_SM))
        btn_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.start_btn = ctk.CTkButton(
            btn_frame, text="▶ 启动服务", font=FONT_BODY, height=38,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#ffffff",
            command=self._start_server,
        )
        self.start_btn.grid(row=0, column=0, padx=PADDING_XS, sticky="ew")

        self.stop_btn = ctk.CTkButton(
            btn_frame, text="⏹ 停止服务", font=FONT_BODY, height=38,
            fg_color=BG_TERTIARY, hover_color=BG_SECONDARY, text_color=FG_PRIMARY,
            command=self._stop_server, state="disabled",
        )
        self.stop_btn.grid(row=0, column=1, padx=PADDING_XS, sticky="ew")

        ctk.CTkButton(
            btn_frame, text="数据预处理", font=FONT_BODY, height=38,
            fg_color=BG_TERTIARY, hover_color=BG_SECONDARY, text_color=FG_PRIMARY,
            command=self._generate_web_data,
        ).grid(row=0, column=2, padx=PADDING_XS, sticky="ew")

        ctk.CTkButton(
            btn_frame, text="打开浏览器", font=FONT_BODY, height=38,
            fg_color=BG_TERTIARY, hover_color=BG_SECONDARY, text_color=FG_PRIMARY,
            command=self._open_browser,
        ).grid(row=0, column=3, padx=PADDING_XS, sticky="ew")

        # ===== 信息提示卡片 =====
        info_card = self._make_card(self.main_frame)
        info_card.grid(row=2, column=0, padx=PADDING, pady=PADDING_XS, sticky="ew")

        ctk.CTkLabel(
            info_card, text="使用说明",
            font=(FONT_BODY[0], FONT_BODY[1], "bold"), text_color=FG_PRIMARY,
        ).pack(anchor="w", padx=PADDING, pady=(PADDING, PADDING_XS))

        tips = [
            "点击「启动服务」启动 FastAPI 后端",
            "点击「打开浏览器」访问 Web 平台 (http://localhost:8000/app)",
            "首次使用需点击「数据预处理」生成 Web 展示数据",
            "API 文档: http://localhost:8000/docs",
        ]
        for tip in tips:
            ctk.CTkLabel(
                info_card, text=f"  ·  {tip}",
                font=FONT_SMALL, text_color=FG_MUTED, anchor="w",
            ).pack(fill="x", padx=(PADDING + 8, PADDING), pady=PADDING_XS)

        ctk.CTkFrame(info_card, fg_color="transparent", height=PADDING_SM).pack()

        # ===== 服务日志卡片 =====
        log_card = self._make_card(self.main_frame)
        log_card.grid(row=3, column=0, padx=PADDING, pady=(PADDING_XS, PADDING), sticky="nsew")
        log_card.grid_columnconfigure(0, weight=1)
        log_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            log_card, text="服务日志", font=FONT_SUBTITLE, text_color=ACCENT,
        ).grid(row=0, column=0, padx=PADDING, pady=(PADDING, PADDING_XS), sticky="w")

        self.log_text = ctk.CTkTextbox(
            log_card, font=FONT_MONO,
            fg_color=BG_CODE, text_color="#94a3b8",
            corner_radius=8, border_width=0,
        )
        self.log_text.grid(row=1, column=0, padx=PADDING, pady=(0, PADDING), sticky="nsew")

    def _log(self, msg: str):
        """输出日志"""
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")
        self.app.update_idletasks()

    def _set_running(self, running: bool):
        """更新运行状态"""
        self.is_running = running
        if running:
            self.status_indicator.configure(text="● 运行中", text_color=ACCENT)
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
        else:
            self.status_indicator.configure(text="● 已停止", text_color=DANGER)
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

    def _start_server(self):
        """启动FastAPI后端"""
        if self.is_running:
            self._log("服务已在运行中")
            return

        port = self.port_var.get().strip()
        if not port.isdigit():
            messagebox.showerror("错误", "端口号必须是数字")
            return

        stats_path = PROJECT_ROOT / "web" / "backend" / "data" / "stats" / "all_stats.json"
        if not stats_path.exists():
            self._log("未检测到预处理数据，建议先点击「数据预处理」")

        self._log(f"正在启动服务 (端口: {port})...")

        def _run():
            try:
                cmd = [
                    sys.executable, "-m", "uvicorn",
                    "web.backend.main:app",
                    "--host", "0.0.0.0",
                    "--port", port,
                ]
                self.server_process = subprocess.Popen(
                    cmd, cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )

                self.app.after(0, lambda: self._set_running(True))
                self.app.after(0, lambda: self._log(f"服务已启动: http://localhost:{port}"))
                self.app.after(0, lambda: self.app.set_status(f"Web服务运行中 (端口: {port})"))

                for line in self.server_process.stdout:
                    line = line.strip()
                    if line:
                        self.app.after(0, lambda l=line: self._log(l))

            except Exception as e:
                self.app.after(0, lambda: self._log(f"启动失败: {e}"))
                self.app.after(0, lambda: self._set_running(False))

        threading.Thread(target=_run, daemon=True).start()

    def _stop_server(self):
        """停止FastAPI后端"""
        if not self.is_running or self.server_process is None:
            return

        self._log("正在停止服务...")
        try:
            self.server_process.terminate()
            self.server_process.wait(timeout=5)
        except Exception:
            try:
                self.server_process.kill()
            except Exception:
                pass

        self.server_process = None
        self._set_running(False)
        self._log("服务已停止")
        self.app.set_status("Web服务已停止")

    def _generate_web_data(self):
        """运行数据预处理脚本"""
        self._log("开始数据预处理 (可能需要几分钟)...")

        def _run():
            try:
                cmd = [sys.executable, "-m", "web.scripts.generate_web_data"]
                process = subprocess.Popen(
                    cmd, cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )

                for line in process.stdout:
                    line = line.strip()
                    if line:
                        self.app.after(0, lambda l=line: self._log(l))

                process.wait()
                if process.returncode == 0:
                    self.app.after(0, lambda: self._log("数据预处理完成"))
                else:
                    self.app.after(0, lambda: self._log(f"数据预处理失败 (退出码: {process.returncode})"))

            except Exception as e:
                self.app.after(0, lambda: self._log(f"预处理出错: {e}"))

        threading.Thread(target=_run, daemon=True).start()

    def _open_browser(self):
        """打开浏览器"""
        port = self.port_var.get().strip() or "8000"
        url = f"http://localhost:{port}/app"
        webbrowser.open(url)
        self._log(f"已打开浏览器: {url}")


if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    root = ctk.CTk()
    root.title("Web 服务管理")
    root.geometry("700x600")
    tab = WebServiceTab(root, root)
    root.mainloop()
