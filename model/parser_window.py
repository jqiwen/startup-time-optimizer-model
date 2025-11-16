import os
import json
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

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

            if field.lower() == "cpu":
                limit_val = limits.get("cpus", limits.get("cpu"))
                resv_val = reservations.get("cpus", reservations.get("cpu"))

            item[field] = {
                "limit": limit_val,
                "reservation": resv_val,
            }

        results.append(item)

    return results


class YamlViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YAML File Parser")
        self.root.geometry("950x600")

        self.selected_file = None
        self.last_structured = None
        self.save_path = None

        self.btn_select = tk.Button(
            root, text="Select YAML File",
            command=self.select_file,
            height=2, width=25
        )
        self.btn_select.pack(pady=(20, 10))

        self.file_label = tk.Label(root, text="No Selected File")
        self.file_label.pack(pady=(0, 10))

        fields_frame = tk.Frame(root)
        fields_frame.pack(pady=(0, 10))

        tk.Label(fields_frame, text="Resource fields (use ',' to seperate ):").pack(side=tk.LEFT)
        self.fields_entry = tk.Entry(fields_frame, width=30)
        self.fields_entry.insert(0, "cpu,memory")
        self.fields_entry.pack(side=tk.LEFT, padx=5)

        self.btn_parse = tk.Button(
            fields_frame, text="PARSE",
            command=self.parse_selected, height=1, width=14
        )
        self.btn_parse.pack(side=tk.LEFT, padx=(10, 0))

        text_frame = tk.Frame(root)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 5))

        self.text_box = scrolledtext.ScrolledText(
            text_frame, wrap=tk.NONE, font=("Consolas", 10)
        )
        self.text_box.pack(fill=tk.BOTH, expand=True)

        self.btn_save = tk.Button(
            root,
            text="Save and Exit",
            command=self.save_and_close,
            state=tk.DISABLED
        )
        self.btn_save.pack(pady=(5, 5))

        self.bottom_label = tk.Label(
            root,
            text="result will be saved to ./data/yaml-parse.json",
            fg="gray"
        )
        self.bottom_label.pack(pady=(0, 10))

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select YAML File",
            filetypes=[("YAML files", "*.yml *.yaml ")]
        )
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ALLOWED_EXT:
            messagebox.showerror("File Type Error", "only support .yml / .yaml File")
            return

        self.selected_file = file_path
        self.file_label.config(text=file_path)
        self.text_box.delete("1.0", tk.END)
        self.last_structured = None
        self.btn_save.config(state=tk.DISABLED)

    def parse_selected(self):
        if not self.selected_file:
            messagebox.showwarning("No Selected File", "Please select a YAML File")
            return

        field_str = self.fields_entry.get().strip()
        resource_fields = [f.strip() for f in field_str.split(",") if f.strip()]

        self.load_and_parse_yaml(self.selected_file, resource_fields)

    def load_and_parse_yaml(self, file_path, resource_fields):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            messagebox.showerror("Parse Error", f"Fail to read of parse YAML: \n{e}")
            return

        results = parse_resources_from_yaml(data, resource_fields)

        self.text_box.delete("1.0", tk.END)

        if not results:
            self.text_box.insert(tk.END, "Not find deploy.resources \n")
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

        # 第一次保存：弹窗选择路径
        if not hasattr(self, "save_path") or self.save_path is None:
            save_path = filedialog.asksaveasfilename(
                title="Choose Save Location",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not save_path:
                return  # 用户取消保存
            self.save_path = save_path
        else:
            save_path = self.save_path

        # 创建目录（如果存在多级路径）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.last_structured, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Failed", f"Save Failed:\n{e}")
            return

        # 成功提示
        messagebox.showinfo("Success", f"Parse result saved to:\n{save_path}")

        self.root.destroy()

        
        

if __name__ == "__main__":
    root = tk.Tk()
    app = YamlViewerApp(root)
    root.mainloop()
