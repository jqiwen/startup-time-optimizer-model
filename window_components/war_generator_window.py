import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from .war_generator import war_generator 
from .popup_success import popup_success

class WarGeneratorWindow:

    def __init__(self, root: tk.Tk, on_next):

        self.root = root
        self.on_next = on_next
        self.frame = tk.Frame(root)

        # --- App name ---
        frame_app = tk.Frame(self.frame)
        frame_app.pack(pady=10)
        frame_app.place(relx=0.5, rely=0.15, anchor="center")

        tk.Label(frame_app, text="Output Location:").pack(side=tk.LEFT)
        self.app_name_entry = tk.Entry(frame_app, width=30)
        self.app_name_entry.insert(0, "local_app")
        self.app_name_entry.config(state="disabled")  
        self.app_name_entry.pack(side=tk.LEFT, padx=5)

        # Info label
        self.info_label = tk.Label(
            self.frame,
            text=(
                "Files will be generated under ./local_env/local_app/"
            ),
            fg="gray",
            wraplength=550,
            justify="left",
        )
        self.info_label.place(relx=0.5, rely=0.20, anchor="center")


        # --- Maven project dir ---
        frame_src = tk.Frame(self.frame)
        frame_src.pack(pady=10)
        frame_src.place(relx=0.5, rely=0.25, anchor="center")

        tk.Label(frame_src, text="Original Project Directory:").pack(side=tk.LEFT)
        self.app_src_entry = tk.Entry(frame_src, width=40)
        self.app_src_entry.insert(0, "./acmeair")
        self.app_src_entry.pack(side=tk.LEFT, padx=5)

        browse_btn = tk.Button(frame_src, text="Browse", command=self.browse_dir)
        browse_btn.pack(side=tk.LEFT)

        # --- Build button ---
        build_btn = tk.Button(
            self.frame,
            text="Build Local Environment",
            command=self.build_and_setup,
            height=2,
            width=25,
        )
        build_btn.place(relx=0.5, rely=0.38, anchor="center")

        # Info label
        self.info_label = tk.Label(
            self.frame,
            text=(
                "Step 1/2: Build the application and generate the environment under ./local_env/local_app/"
            ),
            fg="gray",
            wraplength=550,
            justify="left",
        )
        self.info_label.place(relx=0.5, rely=0.80, anchor="center")

        # --- Next button ---
        self.next_btn = tk.Button(
            self.frame,
            text="Next → Parse YAML",
            command=self.go_next,
            height=1,
            width=20,
            state=tk.DISABLED,
        )
        self.next_btn.place(relx=0.5, rely=0.90, anchor="center")

    def browse_dir(self):
        dir_path = filedialog.askdirectory(
            title="Select Maven project directory (contains pom.xml)"
        )
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
            app_env_dir = war_generator(app_name, app_src_dir)
        except Exception as e:
            messagebox.showerror("Failed to set up app", f"{e}")
            return

        # messagebox.showinfo(
        #     "Success",
        #     f"Environment for '{app_name}' created at:\n{app_env_dir}\n\n"
        #     f"Files:\n  - {app_env_dir / (app_name + '.war')}\n"
        #     f"  - {app_env_dir / 'Dockerfile'}\n"
        #     f"  - {app_env_dir / 'server.xml'}",
        # )
        popup_success(self.root, f"Local Environment has been created under ./local_env/local_app")
        self.next_btn.config(state=tk.NORMAL)

    def go_next(self):
        if self.on_next:
            self.on_next()

