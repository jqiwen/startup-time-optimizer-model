import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from pathlib import Path

from .popup_success import popup_success
from .update_docker_compose import update_docker_compose


class ResourceInputWindow:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.last_structured = None
        self.frame = tk.Frame(root)

        title = tk.Label(self.frame, text="Original Resource Input", font=("Times New Roman", 16, "bold"))
        title.pack(pady=10)

        center_frame = tk.Frame(self.frame)
        center_frame.pack(pady=5)

        button_style = {"width": 15, "height": 3}

        tk.Label(center_frame, text="CPU:", width=10, anchor="e", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=20, pady=5)

        tk.Label(center_frame, text="Limit").grid(row=0, column=1)
        self.cpu_limit = tk.Entry(center_frame, width=12)
        self.cpu_limit.insert(0, "1.0") 
        self.cpu_limit.grid(row=0, column=2, padx=5)

        tk.Label(center_frame, text="Reservation").grid(row=0, column=3)
        self.cpu_resv = tk.Entry(center_frame, width=12)
        self.cpu_resv.insert(0, "0.25") 
        self.cpu_resv.grid(row=0, column=4, padx=5)

        tk.Label(center_frame, text="Memory:", width=10, anchor="e",font=("Arial", 10, "bold")).grid(row=1, column=0, padx=20, pady=5)

        tk.Label(center_frame, text="Limit").grid(row=1, column=1)
        self.mem_limit = tk.Entry(center_frame, width=12)
        self.mem_limit.insert(0,"1G")
        self.mem_limit.grid(row=1, column=2, padx=5)

        tk.Label(center_frame, text="Reservation").grid(row=1, column=3)
        self.mem_resv = tk.Entry(center_frame, width=12)
        self.mem_resv.insert(0,"512M")
        self.mem_resv.grid(row=1, column=4, padx=5)

        self.btn_generate = tk.Button(
            center_frame,
            text="Generate JSON",
            command=self.generate_json,
            **button_style
        )
        self.btn_generate.grid(row=1, column=5, padx=15)

        tk.Label(center_frame, text="Heap:", width=10, anchor="e", font=("Arial", 10, "bold")).grid(row=2, column=0, padx=20, pady=5)

        tk.Label(center_frame, text="Limit").grid(row=2, column=1)
        self.heap_limit = tk.Entry(center_frame, width=12)
        self.heap_limit.insert(0,"1G")
        self.heap_limit.grid(row=2, column=2, padx=5)

        tk.Label(center_frame, text="Reservation").grid(row=2, column=3)
        self.heap_resv = tk.Entry(center_frame, width=12)
        self.heap_resv.insert(0,"512M")
        self.heap_resv.grid(row=2, column=4, padx=5)

        text_frame = tk.Frame(self.frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 5))

        self.text_box = scrolledtext.ScrolledText(
            text_frame, wrap=tk.NONE, font=("Consolas", 10)
        )
        self.text_box.pack(fill=tk.BOTH, expand=True)

        btn_frame = tk.Frame(self.frame)
        btn_frame.pack(pady=10)

        self.btn_save = tk.Button(
            btn_frame,
            text="Save and Run",
            command=self.save_and_run,
            state=tk.DISABLED,
            width=15,
        )
        self.btn_save.pack()


    def generate_json(self):
        cpu_limit = self.cpu_limit.get().strip()
        cpu_resv = self.cpu_resv.get().strip()
        mem_limit = self.mem_limit.get().strip()
        mem_resv = self.mem_resv.get().strip()
        heap_limit = self.heap_limit.get().strip()
        heap_resv = self.heap_resv.get().strip()

        structured = [
            {
                "cpu": {"limit": cpu_limit, "reservation": cpu_resv},
                "memory": {"limit": mem_limit, "reservation": mem_resv},
                "heap": {"limit": heap_limit, "reservation": heap_resv},
            }
        ]

        self.last_structured = structured

        self.text_box.delete("1.0", tk.END)
        self.text_box.insert(
            tk.END,
            json.dumps(structured, indent=2, ensure_ascii=False)
        )

        self.btn_save.config(state=tk.NORMAL)


    def save_and_run(self):
        if not self.last_structured:
            messagebox.showwarning("No Data", "Please generate JSON first.")
            return

        parent_path = Path(__file__).resolve().parent.parent
        save_path = parent_path / "local_env" / "resource_input.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.last_structured, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Failed", f"Save Failed:\n{e}")
            return
        
        update_docker_compose(self.last_structured, compose_path="./local_env/docker-compose.yml")

        popup_success(
            self.root,
            "Resources Updated",
            on_close=self.root.destroy
        )
