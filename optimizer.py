import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Please install PyYAML first: pip install pyyaml")
    sys.exit(1)

from window_components.war_generator_window import WarGeneratorWindow 
from window_components.resource_input_window import ResourceInputWindow

from window_components.popup_success import popup_success

def main():
    root = tk.Tk()
    root.title("Startup Time Optimizer")
    root.geometry("950x650")

    setup_step = None
    yaml_step = None

    def show_setup():
        yaml_step.frame.pack_forget()
        setup_step.frame.pack(fill=tk.BOTH, expand=True)

    def show_yaml():
        setup_step.frame.pack_forget()
        yaml_step.frame.pack(fill=tk.BOTH, expand=True)

    setup_step = WarGeneratorWindow(root, on_next=show_yaml)
    yaml_step = ResourceInputWindow(root)

    setup_step.frame.pack(fill=tk.BOTH, expand=True)

    root.mainloop()


if __name__ == "__main__":
    main()
