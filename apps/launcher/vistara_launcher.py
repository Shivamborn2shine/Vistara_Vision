from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import scrolledtext

VERCEL_URL = "https://vistara-vision.vercel.app/?api=http://127.0.0.1:8765"
LOCAL_HEALTH_URL = "http://127.0.0.1:8765/api/v1/health"
PUBLIC_URL_ENDPOINT = "http://127.0.0.1:8765/api/v1/pairing/public-url"
REPO_ROOT = Path(__file__).resolve().parents[2]
TRYCLOUDFLARE_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


def find_python_executable() -> str:
    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def find_cloudflared_executable() -> str | None:
    venv_cf = REPO_ROOT / ".venv" / "Scripts" / "cloudflared.exe"
    if venv_cf.exists():
        return str(venv_cf)
    return shutil.which("cloudflared")


class VistaraLauncherApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Vistara Vision Launcher")
        self.root.geometry("620x440")
        self.root.configure(bg="#0b1120")

        self.process: subprocess.Popen[str] | None = None
        self.tunnel_process: subprocess.Popen[str] | None = None
        self.tunnel_url: str | None = None
        self._stopping = False

        header = tk.Label(
            root,
            text="Vistara Vision — Local Backend & Web Launcher",
            font=("Segoe UI", 14, "bold"),
            fg="#f8fafc",
            bg="#0b1120",
        )
        header.pack(pady=(14, 4))

        self.status_var = tk.StringVar(value="Status: Backend Stopped")
        status_lbl = tk.Label(
            root,
            textvariable=self.status_var,
            font=("Segoe UI", 11, "bold"),
            fg="#38bdf8",
            bg="#0b1120",
        )
        status_lbl.pack(pady=(0, 10))

        btn_frame = tk.Frame(root, bg="#0b1120")
        btn_frame.pack(pady=6)

        self.start_btn = tk.Button(
            btn_frame,
            text="Start Backend & Open Vistara Vision",
            font=("Segoe UI", 10, "bold"),
            bg="#38bdf8",
            fg="#06121f",
            padx=14,
            pady=8,
            command=self.start_and_launch,
        )
        self.start_btn.grid(row=0, column=0, padx=6, pady=4)

        self.open_btn = tk.Button(
            btn_frame,
            text="Open Vercel URL",
            font=("Segoe UI", 10, "bold"),
            bg="#1e293b",
            fg="#f8fafc",
            padx=12,
            pady=8,
            command=lambda: webbrowser.open(VERCEL_URL),
        )
        self.open_btn.grid(row=0, column=1, padx=6, pady=4)

        self.stop_btn = tk.Button(
            btn_frame,
            text="Stop Backend",
            font=("Segoe UI", 10, "bold"),
            bg="#ef4444",
            fg="#ffffff",
            padx=12,
            pady=8,
            command=self.stop_backend,
        )
        self.stop_btn.grid(row=0, column=2, padx=6, pady=4)

        url_lbl = tk.Label(
            root,
            text=f"Target URL: {VERCEL_URL}",
            font=("Segoe UI", 9),
            fg="#cbd5e1",
            bg="#0b1120",
        )
        url_lbl.pack(pady=(4, 6))

        self.log_box = scrolledtext.ScrolledText(
            root,
            height=14,
            font=("Consolas", 9),
            bg="#11192c",
            fg="#e2e8f0",
            insertbackground="#f8fafc",
        )
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=14, pady=(4, 14))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        if self._is_backend_healthy():
            self.status_var.set("Status: Backend Already Running (127.0.0.1:8765)")
            self.log("Detected Vistara backend already running on http://127.0.0.1:8765.")

    def log(self, message: str) -> None:
        def _append() -> None:
            self.log_box.insert(tk.END, message.rstrip() + "\n")
            self.log_box.see(tk.END)

        self.root.after(0, _append)

    @staticmethod
    def _is_backend_healthy() -> bool:
        try:
            with urllib.request.urlopen(LOCAL_HEALTH_URL, timeout=1.0) as resp:
                return bool(resp.status == 200)
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def start_and_launch(self) -> None:
        if self._is_backend_healthy():
            self.status_var.set("Status: Backend Running (127.0.0.1:8765)")
            self.log("Backend is already healthy. Ensuring Cloudflare tunnel & opening browser...")
            self._start_cloudflared()
            webbrowser.open(VERCEL_URL)
            return

        if self.process is not None and self.process.poll() is None:
            self.log("Backend is starting... please wait.")
            return

        py_exe = find_python_executable()
        cmd = [
            py_exe,
            "-m",
            "uvicorn",
            "services.core.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8765",
        ]
        self.status_var.set("Status: Starting Backend...")
        self.log(f"Starting: {' '.join(cmd)}")
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creation_flags,
        )

        threading.Thread(target=self._stream_logs, daemon=True).start()
        threading.Thread(target=self._wait_and_open_browser, daemon=True).start()
        self._start_cloudflared()

    def _start_cloudflared(self) -> None:
        if self.tunnel_process is not None and self.tunnel_process.poll() is None:
            return
        cf_exe = find_cloudflared_executable()
        if not cf_exe:
            self.log("cloudflared not found; phone pairing will use local Wi-Fi IP.")
            return
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.log("Starting Cloudflare Tunnel for internet phone pairing...")
        self.tunnel_process = subprocess.Popen(
            [cf_exe, "tunnel", "--url", "http://127.0.0.1:8765"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creation_flags,
        )
        threading.Thread(target=self._stream_tunnel_logs, daemon=True).start()

    def _stream_tunnel_logs(self) -> None:
        if self.tunnel_process is None or self.tunnel_process.stdout is None:
            return
        for line in self.tunnel_process.stdout:
            m = TRYCLOUDFLARE_RE.search(line)
            if m and not self.tunnel_url:
                self.tunnel_url = m.group(0)
                self.log(f"Cloudflare Tunnel Ready: {self.tunnel_url}")
                self._push_public_url(self.tunnel_url)
            elif "ERR" in line or "Registered tunnel" in line:
                self.log(f"[tunnel] {line.strip()}")

    def _push_public_url(self, url: str | None) -> None:
        for _ in range(15):
            if self._is_backend_healthy():
                break
            time.sleep(0.5)
        try:
            body = json.dumps({"public_url": url}).encode("utf-8")
            req = urllib.request.Request(
                PUBLIC_URL_ENDPOINT,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2.0):
                pass
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.log(f"Could not register tunnel URL with backend: {exc}")

    def _stream_logs(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            self.log(line)
        if not self._stopping:
            self.root.after(0, lambda: self.status_var.set("Status: Backend Stopped"))

    def _wait_and_open_browser(self) -> None:
        for _ in range(30):
            if self._stopping:
                return
            if self._is_backend_healthy():
                self.root.after(
                    0, lambda: self.status_var.set("Status: Backend Running (127.0.0.1:8765)")
                )
                self.log("Backend ready! Launching https://vistara-vision.vercel.app/ ...")
                webbrowser.open(VERCEL_URL)
                return
            time.sleep(0.5)
        self.log("Timed out waiting for backend health check on port 8765.")

    def stop_backend(self) -> None:
        if self.tunnel_process is not None and self.tunnel_process.poll() is None:
            self.tunnel_process.terminate()
            self.log("Stopped Cloudflare Tunnel.")
        self.tunnel_process = None
        self.tunnel_url = None
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.log("Stopped local Vistara backend.")
        self.process = None
        if os.name == "nt":
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue | "
                    "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }",
                ],
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
        self.status_var.set("Status: Backend Stopped")

    def on_close(self) -> None:
        self._stopping = True
        self.stop_backend()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    VistaraLauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
