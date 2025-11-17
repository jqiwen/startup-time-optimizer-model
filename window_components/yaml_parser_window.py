import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from pathlib import Path

from .popup_success import popup_success

try:
    import yaml
except ImportError:
    print("Please install PyYAML first: pip install pyyaml")
    sys.exit(1)


ALLOWED_EXT = {".yml", ".yaml"}

def parse_resources_from_yaml(data, resource_fields):

    results = []

    if not isinstance(data, dict):
        return results

    services = data.get("services", {})
    if not isinstance(services, dict):
        return results

    for svc_name, svc_conf in services.items():
        deploy = svc_conf.get("deploy", {})
        resources = deploy.get("resources", {})
        limits = resources.get("limits", {}) or {}
        reservations = resources.get("reservations", {}) or {}

        item = {"service": svc_name}

        for raw_field in resource_fields:
            field = raw_field.strip()
            if not field:
                continue

            limit_val = limits.get(field)
            resv_val = reservations.get(field)

            # specific cpu / cpus
            if field.lower() == "cpu":
                limit_val = limits.get("cpus", limits.get("cpu"))
                resv_val = reservations.get("cpus", reservations.get("cpu"))

            item[field] = {
                "limit": limit_val,
                "reservation": resv_val,
            }

        results.append(item)

    return results

class YamlParserWindow:

    def __init__(self, root: tk.Tk):

        self.root = root
        # self.on_back = on_back

        self.selected_file = None
        self.last_structured = None
        self.save_path = None

        self.frame = tk.Frame(root)

        # select file button
        top_area = tk.Frame(self.frame)
        top_area.pack(fill=tk.X, pady=(15, 5))
        self.btn_select = tk.Button(
            top_area,
            text="Select YAML File",
            command=self.select_file,
            height=1,
            width=20,
        )
        self.btn_select.pack(pady=5)

        # file path
        self.file_label = tk.Label(top_area, text="No Selected File")
        self.file_label.pack()

        # resource fields 
        fields_frame = tk.Frame(self.frame)
        fields_frame.pack(pady=(5, 10))

        tk.Label(fields_frame, text="Resource fields (use ';' to separate):").pack(side=tk.LEFT)
        self.fields_entry = tk.Entry(fields_frame, width=30)
        self.fields_entry.insert(0, "cpu;memory")
        self.fields_entry.pack(side=tk.LEFT, padx=5)

        self.btn_parse = tk.Button(
            fields_frame,
            text="PARSE",
            command=self.parse_selected,
            height=1,
            width=14,
        )
        self.btn_parse.pack(side=tk.LEFT, padx=(10, 0))

        # display content area
        text_frame = tk.Frame(self.frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 5))

        self.text_box = scrolledtext.ScrolledText(
            text_frame, wrap=tk.NONE, font=("Consolas", 10)
        )
        self.text_box.pack(fill=tk.BOTH, expand=True)

        # bottom button
        bottom_frame = tk.Frame(self.frame)
        bottom_frame.pack(pady=(5, 5))

        # info 
        self.bottom_label = tk.Label(
            bottom_frame, 
            text="Step 2/2: The parsing result will be saved under ./local_env/yaml_parser_results.json",
            fg="gray",
        )
        self.bottom_label.pack(pady=(0, 5))

        self.btn_save = tk.Button(
            bottom_frame,
            text="Save and Run",
            command=self.save_and_close,
            state=tk.DISABLED,
            width=15,
        )
        self.btn_save.pack()
        



    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select YAML File",
            filetypes=[("YAML files", "*.yml *.yaml")],
        )
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ALLOWED_EXT:
            messagebox.showerror("File Type Error", "Only support .yml / .yaml files.")
            return

        self.selected_file = file_path
        self.file_label.config(text=file_path)
        self.text_box.delete("1.0", tk.END)
        self.last_structured = None
        self.btn_save.config(state=tk.DISABLED)

    def parse_selected(self):
        if not self.selected_file:
            messagebox.showwarning("No Selected File", "Please select a YAML file.")
            return

        field_str = self.fields_entry.get().strip()
        resource_fields = [f.strip() for f in field_str.split(";") if f.strip()]

        self.load_and_parse_yaml(self.selected_file, resource_fields)

    def load_and_parse_yaml(self, file_path, resource_fields):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            messagebox.showerror("Parse Error", f"Fail to read or parse YAML:\n{e}")
            return

        results = parse_resources_from_yaml(data, resource_fields)

        self.text_box.delete("1.0", tk.END)

        if not results:
            self.text_box.insert(tk.END, "Not find deploy.resources\n")
            self.btn_save.config(state=tk.DISABLED)
            return

        structured = []
        for item in results:
            svc = item["service"]
            fields = {k: v for k, v in item.items() if k != "service"}
            structured.append({svc: fields})

        self.last_structured = structured
        self.btn_save.config(state=tk.NORMAL)
        self.text_box.insert(tk.END, json.dumps(structured, indent=2, ensure_ascii=False))

    def save_and_close(self):
        if not self.last_structured:
            messagebox.showwarning("No Data", "Please parse a file first.")
            return

        # if not self.save_path:
        #     save_path = filedialog.asksaveasfilename(
        #         title="Choose Save Location",
        #         defaultextension=".json",
        #         filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        #     )
        #     if not save_path:
        #         return
        #     self.save_path = save_path
        # else:
        #     save_path = self.save_path

        parent_path = Path(__file__).resolve().parent.parent
        save_path = parent_path / "local_env" / "yaml_parser_results.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.last_structured, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Failed", f"Save Failed:\n{e}")
            return

        popup_success(self.root, f"File has been created under ./local_env/yaml_parser_results.json",on_close=self.root.destroy)

        # self.root.destroy()
