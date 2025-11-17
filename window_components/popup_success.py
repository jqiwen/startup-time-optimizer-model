import tkinter as tk

def popup_success(root, message: str, on_close=None):

    popup = tk.Toplevel(root)
    popup.title("Success")
    popup.resizable(False, False)


    popup.update_idletasks()
    w, h = 420, 220
    x = root.winfo_x() + (root.winfo_width() - w) // 2
    y = root.winfo_y() + (root.winfo_height() - h) // 2
    popup.geometry(f"{w}x{h}+{x}+{y}")

    popup.transient(root)
    popup.grab_set()

    top_frame = tk.Frame(popup)
    top_frame.pack(pady=15)

    icon_label = tk.Label(
        top_frame,
        text="✔",
        fg="#2ecc71",           
        font=("Segoe UI", 36, "bold")
    )
    icon_label.pack(side=tk.LEFT, padx=(0, 10))

    title_label = tk.Label(
        top_frame,
        text="Success",
        font=("Segoe UI", 18, "bold"),
    )
    title_label.pack(side=tk.LEFT)

    msg_label = tk.Label(
        popup,
        text=message,
        font=("Segoe UI", 11),
        justify="left",
        wraplength=380,
        anchor="w"
    )
    msg_label.pack(padx=20, pady=(0, 15), fill="x")

    def close_popup():
        popup.destroy()
        if on_close:
            on_close()

    btn = tk.Button(
        popup,
        text="OK",
        width=10,
        command=close_popup
    )
    btn.pack(pady=(0, 15))

    btn.focus_set()
    popup.bind("<Return>", lambda e: popup.destroy())
