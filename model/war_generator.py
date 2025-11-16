#!/usr/bin/env python3
"""
parser.py

Simple Tkinter GUI frontend for setup_app.py:

1) User types an app name (e.g. 'acmeair')
2) User browses to the Maven project directory (the folder that contains pom.xml)
3) Clicks "Build WAR & Setup App"
4) We run Maven in Docker, find the WAR, and create ../local_env/<app_name>/
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from setup_app import setup_app  # must be in the same folder (model/)


class SetupAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Startup Time Optimizer - App Setup")
        self.root.geometry("600x200")

        # --- App name ---
        frame_app = tk.Frame(root)
        frame_app.pack(pady=10, padx=10, fill=tk.X)

        tk.Label(frame_app, text="App Name:").pack(side=tk.LEFT)
        self.app_name_entry = tk.Entry(frame_app, width=30)
        self.app_name_entry.insert(0, "acmeair")
        self.app_name_entry.pack(side=tk.LEFT, padx=5)

        # --- Maven project dir ---
        frame_src = tk.Frame(root)
        frame_src.pack(pady=10, padx=10, fill=tk.X)

        tk.Label(frame_src, text="Maven Project Directory:").pack(side=tk.LEFT)
        self.app_src_entry = tk.Entry(frame_src, width=40)
        self.app_src_entry.insert(0, "./acmeair")
        self.app_src_entry.pack(side=tk.LEFT, padx=5)

        browse_btn = tk.Button(frame_src, text="Browse", command=self.browse_dir)
        browse_btn.pack(side=tk.LEFT)

        # --- Build button ---
        build_btn = tk.Button(
            root,
            text="Build WAR & Setup App",
            command=self.build_and_setup,
            height=2,
            width=25,
        )
        build_btn.pack(pady=15)

        # Info label
        self.info_label = tk.Label(
            root,
            text="This will run 'mvn -DskipTests package' inside Docker and create ../local_env/<app_name>/",
            fg="gray",
            wraplength=550,
            justify="left",
        )
        self.info_label.pack(padx=10)

    def browse_dir(self):
        dir_path = filedialog.askdirectory(title="Select Maven project directory (contains pom.xml)")
        if dir_path:
            self.app_src_entry.delete(0, tk.END)
            self.app_src_entry.insert(0, dir_path)

    def build_and_setup(self):
        app_name = self.app_name_entry.get().strip()
        app_src = self.app_src_entry.get().strip()

        if not app_name:
            messagebox.showwarning("Missing app name", "Please enter an application name.")
            return
        if not app_src:
            messagebox.showwarning("Missing source path", "Please choose the Maven project directory.")
            return

        app_src_dir = Path(app_src).expanduser()

        if not app_src_dir.is_dir():
            messagebox.showerror("Invalid directory", f"{app_src_dir} is not a directory.")
            return

        try:
            app_env_dir = setup_app(app_name, app_src_dir)
        except Exception as e:
            messagebox.showerror("Failed to set up app", f"{e}")
            return

        messagebox.showinfo(
            "Success",
            f"Environment for '{app_name}' created at:\n{app_env_dir}\n\n"
            f"Files:\n  - {app_env_dir / (app_name + '.war')}\n"
            f"  - {app_env_dir / 'Dockerfile'}\n"
            f"  - {app_env_dir / 'server.xml'}",
        )


if __name__ == "__main__":
    root = tk.Tk()
    gui = SetupAppGUI(root)
    root.mainloop()
