"""
Landsat数据下载Tab
==================
从landsat_download/main.py重构为CTkFrame可嵌入组件
"""

import os
import json
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import customtkinter as ctk
from tkinter import filedialog, messagebox

from gui.styles import (
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BG_INPUT, BG_CODE,
    FG_PRIMARY, FG_SECONDARY, FG_MUTED,
    ACCENT, ACCENT_HOVER, ACCENT_LIGHT,
    FONT_SUBTITLE, FONT_BODY, FONT_SMALL, FONT_MONO,
    PADDING, PADDING_SM, PADDING_XS, CARD_RADIUS, CREDENTIALS_DIR, BORDER, PROGRESS_HEIGHT,
)

CREDENTIALS_FILE = CREDENTIALS_DIR / "usgs.json"


def _load_credentials():
    """加载USGS凭据"""
    if CREDENTIALS_FILE.exists():
        try:
            with open(CREDENTIALS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"usgs_user": "", "usgs_token": "", "save_dir": "./data/landsat"}


def _save_credentials(config: dict):
    """保存USGS凭据"""
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(CREDENTIALS_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


class LandsatTab:
    """Landsat下载Tab页"""

    def __init__(self, parent: ctk.CTkFrame, app: ctk.CTk):
        self.parent = parent
        self.app = app
        self.downloading = False
        self.config = _load_credentials()
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
            title_frame, text="Landsat 数据下载器",
            font=FONT_SUBTITLE, text_color=ACCENT,
        ).pack(side="left")

        ctk.CTkLabel(
            title_frame, text="  通过 USGS M2M API 下载 Landsat 地表反射率波段",
            font=FONT_SMALL, text_color=FG_MUTED,
        ).pack(side="left", padx=PADDING_SM)

        # ===== 配置区卡片 =====
        config_card = self._make_card(self.main_frame)
        config_card.grid(row=1, column=0, padx=PADDING, pady=PADDING_XS, sticky="ew")
        config_card.grid_columnconfigure(1, weight=1)

        # USGS账号
        self._make_label(config_card, "USGS 账号:", width=110).grid(
            row=0, column=0, padx=(PADDING, PADDING_SM), pady=PADDING_SM, sticky="w")
        self.user_entry = self._make_entry(config_card)
        self.user_entry.grid(row=0, column=1, padx=(PADDING_SM, PADDING), pady=PADDING_SM, sticky="ew")
        self.user_entry.insert(0, self.config.get("usgs_user", ""))

        # API Token
        self._make_label(config_card, "API Token:", width=110).grid(
            row=1, column=0, padx=(PADDING, PADDING_SM), pady=PADDING_XS, sticky="w")
        self.token_entry = self._make_entry(config_card, show="*")
        self.token_entry.grid(row=1, column=1, padx=(PADDING_SM, PADDING), pady=PADDING_XS, sticky="ew")
        self.token_entry.insert(0, self.config.get("usgs_token", ""))

        # 提示
        ctk.CTkLabel(
            config_card,
            text="获取Token: 登录 https://ers.cr.usgs.gov/profile/access 申请M2M API访问",
            font=FONT_SMALL, text_color=FG_MUTED,
        ).grid(row=2, column=0, columnspan=2, padx=PADDING, pady=(0, PADDING_XS), sticky="w")

        # 记住凭据
        self.save_cred_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            config_card, text="记住凭据", variable=self.save_cred_var, font=FONT_BODY,
            text_color=FG_PRIMARY, checkbox_width=20, checkbox_height=20,
        ).grid(row=3, column=0, columnspan=2, padx=PADDING, pady=(0, PADDING_SM), sticky="w")

        # 产品ID
        self._make_label(config_card, "产品ID:", width=110).grid(
            row=4, column=0, padx=(PADDING, PADDING_SM), pady=PADDING_XS, sticky="w")
        self.product_entry = self._make_entry(config_card)
        self.product_entry.grid(row=4, column=1, padx=(PADDING_SM, PADDING), pady=PADDING_XS, sticky="ew")
        self.product_entry.insert(0, "LC08_L2SP_130044_20200516_20200820_02_T1")

        # 保存目录
        self._make_label(config_card, "保存目录:", width=110).grid(
            row=5, column=0, padx=(PADDING, PADDING_SM), pady=(PADDING_XS, PADDING_SM), sticky="w")
        dir_frame = ctk.CTkFrame(config_card, fg_color="transparent")
        dir_frame.grid(row=5, column=1, padx=(PADDING_SM, PADDING), pady=(PADDING_XS, PADDING_SM), sticky="ew")
        dir_frame.grid_columnconfigure(0, weight=1)
        self.dir_entry = self._make_entry(dir_frame)
        self.dir_entry.grid(row=0, column=0, sticky="ew", padx=(0, PADDING_XS))
        self.dir_entry.insert(0, self.config.get("save_dir", "./data/landsat"))
        ctk.CTkButton(
            dir_frame, text="浏览", width=60, height=36, font=FONT_BODY,
            fg_color=BG_TERTIARY, hover_color=BG_SECONDARY, text_color=FG_PRIMARY,
            command=self._select_dir,
        ).grid(row=0, column=1)

        # ===== 操作栏 =====
        action_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        action_frame.grid(row=2, column=0, padx=PADDING, pady=PADDING_SM, sticky="ew")

        self.download_btn = ctk.CTkButton(
            action_frame, text="开始下载", width=150, height=40, font=FONT_BODY,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#ffffff",
            command=self._start_download,
        )
        self.download_btn.pack(side="left")

        self.progress = ctk.CTkProgressBar(action_frame, width=400, height=PROGRESS_HEIGHT,
                                           fg_color=BG_TERTIARY, progress_color=ACCENT)
        self.progress.pack(side="left", padx=PADDING, fill="x", expand=True)
        self.progress.set(0)

        # ===== 日志区 =====
        log_card = self._make_card(self.main_frame)
        log_card.grid(row=3, column=0, padx=PADDING, pady=(PADDING_XS, PADDING), sticky="nsew")
        log_card.grid_columnconfigure(0, weight=1)
        log_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            log_card, text="下载日志", font=FONT_SUBTITLE, text_color=ACCENT,
        ).grid(row=0, column=0, padx=PADDING, pady=(PADDING, PADDING_XS), sticky="w")

        self.log_textbox = ctk.CTkTextbox(
            log_card, font=FONT_MONO,
            fg_color=BG_CODE, text_color="#94a3b8",
            corner_radius=8, state="disabled", border_width=0,
        )
        self.log_textbox.grid(row=1, column=0, padx=PADDING, pady=(0, PADDING), sticky="nsew")

    def _select_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.dir_entry.delete(0, "end")
            self.dir_entry.insert(0, dir_path)

    def _log(self, msg: str):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", msg + "\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")
        self.app.update_idletasks()

    def _filter_sr_downloads(self, secondary_downloads):
        sr_files = []
        st_files = []
        for sd in secondary_downloads:
            name = sd.get("displayId", "")
            if "_ST_" in name.upper():
                st_files.append(name)
                continue
            sr_files.append(sd)
        if st_files:
            self._log(f"  跳过 {len(st_files)} 个地表温度(ST)文件")
        return sr_files

    def _resolve_band_links(self, api, band_downloads, label):
        download_list = [
            {"entityId": sd["entityId"], "productId": sd["id"]}
            for sd in band_downloads
        ]
        result = api.request("download-request", {"downloads": download_list, "label": label})

        valid_ids = set(str(x) for x in result.get("newRecords", [])) | \
                    set(str(x) for x in result.get("duplicateProducts", []))
        n_requested = len(download_list)
        n_failed = len(result.get("failed", []))

        links = []

        def _collect_from_retrieve(retrieve):
            collected = []
            for dl in retrieve.get("available", []) + retrieve.get("requested", []):
                if str(dl["downloadId"]) in valid_ids:
                    collected.append({"entityId": dl["entityId"], "url": dl["url"]})
            return collected

        if result.get("preparingDownloads"):
            import time
            retrieve = api.request("download-retrieve", {"label": label})
            links.extend(_collect_from_retrieve(retrieve))
            seen_ids = {l["url"] for l in links}

            max_retries = 100
            retry_delay = 10
            for attempt in range(max_retries):
                if len(links) >= n_requested - n_failed:
                    break
                n_pending = n_requested - n_failed - len(links)
                self._log(f"  等待下载链接准备就绪... ({n_pending}个待处理, {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retrieve = api.request("download-retrieve", {"label": label})
                for dl in retrieve.get("available", []):
                    if dl["url"] not in seen_ids:
                        links.append({"entityId": dl["entityId"], "url": dl["url"]})
                        seen_ids.add(dl["url"])
        else:
            for dl in result.get("availableDownloads", []):
                links.append({"entityId": dl["entityId"], "url": dl["url"]})

        return links

    def _start_download(self):
        if self.downloading:
            return

        user = self.user_entry.get().strip()
        token = self.token_entry.get().strip()
        product_id = self.product_entry.get().strip()
        save_dir = self.dir_entry.get().strip()

        if not user or not token or not product_id or not save_dir:
            messagebox.showerror("错误", "所有字段都不能为空")
            return

        if self.save_cred_var.get():
            self.config["usgs_user"] = user
            self.config["usgs_token"] = token
            self.config["save_dir"] = save_dir
            _save_credentials(self.config)

        os.makedirs(save_dir, exist_ok=True)

        self.downloading = True
        self.download_btn.configure(state="disabled", text="下载中...")
        self.progress.set(0)
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

        threading.Thread(
            target=self._download_task,
            args=(user, token, product_id, save_dir),
            daemon=True,
        ).start()

    def _download_task(self, user, token, product_id, save_dir):
        import requests
        import time
        from usgsxplore.api import API

        api = None
        try:
            self._log("正在登录 USGS M2M API...")
            api = API(username=user, token=token)
            self._log("登录成功！")

            self._log(f"正在查询产品：{product_id}")
            self.app.after(0, lambda: self.progress.set(0.1))

            dataset = "landsat_ot_c2_l2"
            entity_id = api.get_entity_id(product_id, dataset=dataset)
            self._log(f"实体ID：{entity_id}")
            self.app.after(0, lambda: self.progress.set(0.2))

            raw_options = api.request(
                "download-options", {"datasetName": dataset, "entityIds": [entity_id]},
            )

            all_secondary = []
            for opt in raw_options:
                all_secondary.extend(opt.get("secondaryDownloads", []))

            if not all_secondary:
                raise Exception("未找到可单独下载的波段文件，请检查产品ID是否正确")

            sr_downloads = self._filter_sr_downloads(all_secondary)
            total_mb = sum(sd.get("filesize", 0) for sd in sr_downloads) / 1e6
            self._log(f"筛选出 {len(sr_downloads)} 个地表反射率(SR)文件，共 {total_mb:.0f} MB")
            self._log("跳过所有地表温度(ST)文件，节省约 485MB 下载量")
            self.app.after(0, lambda: self.progress.set(0.3))

            self._log("正在获取下载链接...")
            label = api.label
            links = self._resolve_band_links(api, sr_downloads, label)
            self._log(f"获取到 {len(links)} 个下载链接")
            self.app.after(0, lambda: self.progress.set(0.4))

            eid_to_name = {sd["entityId"]: sd["displayId"] for sd in sr_downloads}
            output_dir = Path(save_dir) / product_id
            output_dir.mkdir(parents=True, exist_ok=True)
            self._log(f"文件将保存到：{output_dir}")

            n_links = len(links)
            completed = [0]
            lock = threading.Lock()

            def _download_one(idx, link):
                filename = eid_to_name.get(link["entityId"], f"band_{idx + 1}.TIF")
                file_path = output_dir / filename
                self._log(f"开始下载 ({idx + 1}/{n_links})：{filename}")

                with requests.get(link["url"], stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            f.write(chunk)

                size_mb = os.path.getsize(file_path) / 1e6
                with lock:
                    completed[0] += 1
                    overall = 0.4 + 0.55 * (completed[0] / n_links)
                    self.app.after(0, lambda p=min(overall, 0.95): self.progress.set(p))
                self._log(f"  已保存 ({completed[0]}/{n_links})：{filename} ({size_mb:.1f} MB)")

            max_workers = min(n_links, 8)
            self._log(f"使用 {max_workers} 个线程并发下载 {n_links} 个文件...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_download_one, i, link): i for i, link in enumerate(links)}
                for future in as_completed(futures):
                    future.result()

            try:
                api.request("download-order-remove", {"label": label})
            except Exception:
                pass

            self.app.after(0, lambda: self.progress.set(1))
            self._log(f"\n下载完成！文件保存在：{output_dir}")
            self._log(f"共下载 {n_links} 个地表反射率(SR)波段文件，已跳过所有地表温度(ST)文件！")
            self.app.after(0, lambda: messagebox.showinfo("成功", "下载完成！"))

        except Exception as e:
            self._log(f"出错：{str(e)}")
            self.app.after(0, lambda: messagebox.showerror("错误", f"下载失败：{str(e)}"))
        finally:
            if api:
                try:
                    api.logout()
                except Exception:
                    pass
            self.downloading = False
            self.app.after(0, lambda: self.download_btn.configure(state="normal", text="开始下载"))


if __name__ == "__main__":
    import customtkinter as ctk
    ctk.set_appearance_mode("light")
    root = ctk.CTk()
    root.title("Landsat 数据下载器")
    root.geometry("750x700")
    tab = LandsatTab(root, root)
    root.mainloop()
