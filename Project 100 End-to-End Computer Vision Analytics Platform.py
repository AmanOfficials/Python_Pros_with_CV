import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk
from ultralytics import YOLO

class EnterpriseVisionPlatform:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 100: End-to-End Vision Analytics Platform")
        self.root.geometry("1200x800")

        self.model = YOLO("yolov8n.pt")
        self.cap = None
        self.is_running = False

        # Top Executive Header
        header = ttk.Frame(self.root, padding=10)
        header.pack(side=tk.TOP, fill=tk.X)

        self.btn_toggle = ttk.Button(header, text="Initialize Camera Feed", command=self.toggle_feed)
        self.btn_toggle.pack(side=tk.LEFT, padx=5)

        self.lbl_system = ttk.Label(header, text="System: IDLE", font=("Arial", 11, "bold"), foreground="gray")
        self.lbl_system.pack(side=tk.LEFT, padx=15)

        ttk.Button(header, text="Export Audit Log", command=self.export_logs).pack(side=tk.RIGHT, padx=5)

        # Main Workspace: Left Viewport (70%), Right Metrics Dashboard (30%)
        workspace = ttk.Frame(self.root)
        workspace.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.video_panel = ttk.Label(workspace)
        self.video_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right Telemetry Sidebar
        sidebar = ttk.Frame(workspace, width=350, padding=8)
        sidebar.pack(side=tk.RIGHT, fill=tk.BOTH)

        ttk.Label(sidebar, text="Live Entity Telemetry", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=5)
        
        self.tree = ttk.Treeview(sidebar, columns=("Class", "Confidence", "Timestamp"), show="headings", height=18)
        self.tree.heading("Class", text="Entity")
        self.tree.column("Class", width=90)
        self.tree.heading("Confidence", text="Confidence")
        self.tree.column("Confidence", width=80)
        self.tree.heading("Timestamp", text="Time")
        self.tree.column("Timestamp", width=120)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Bottom Diagnostic Footer
        self.footer = ttk.Frame(self.root, padding=6)
        self.footer.pack(side=tk.BOTTOM, fill=tk.X)
        self.lbl_fps = ttk.Label(self.footer, text="FPS: 0.0")
        self.lbl_fps.pack(side=tk.LEFT, padx=10)
        self.lbl_total = ttk.Label(self.footer, text="Total Detected: 0")
        self.lbl_total.pack(side=tk.LEFT, padx=10)

    def toggle_feed(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn_toggle.configure(text="Terminate Feed")
                self.lbl_system.configure(text="System: STREAMING & LOGGING", foreground="green")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn_toggle.configure(text="Initialize Camera Feed")
            self.lbl_system.configure(text="System: IDLE", foreground="gray")

    def loop(self):
        if not self.is_running:
            return

        t_start = cv2.getTickCount()
        ret, frame = self.cap.read()
        if ret:
            results = self.model.track(frame, persist=True, verbose=False)
            boxes = results[0].boxes
            current_time = datetime.now().strftime("%H:%M:%S")

            for box in boxes:
                cls_name = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                # Append telemetry to table
                if len(self.tree.get_children()) > 40:
                    self.tree.delete(self.tree.get_children()[-1])
                self.tree.insert("", 0, values=(cls_name, f"{conf:.2f}", current_time))

            t_end = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (t_end - t_start + 1e-5)

            self.lbl_fps.configure(text=f"FPS: {fps:.1f}")
            self.lbl_total.configure(text=f"Active Tracked Entities: {len(boxes)}")

            annotated = results[0].plot()
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((780, 580))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.video_panel.configure(image=self.tk_photo)

        self.root.after(15, self.loop)

    def export_logs(self):
        records = [self.tree.item(item)["values"] for item in self.tree.get_children()]
        if not records:
            messagebox.showinfo("Export Logs", "No telemetry records to export.")
            return
        with open("vision_audit_log.csv", "w") as f:
            f.write("Entity,Confidence,Timestamp\n")
            for r in records:
                f.write(f"{r[0]},{r[1]},{r[2]}\n")
        messagebox.showinfo("Export Logs", "Audit log exported to vision_audit_log.csv")

if __name__ == "__main__":
    root = tk.Tk()
    app = EnterpriseVisionPlatform(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()