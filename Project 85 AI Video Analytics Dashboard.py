import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

class AnalyticsDashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 85: AI Video Analytics Dashboard")
        self.root.geometry("1100x700")

        self.model = YOLO("yolov8n.pt")
        self.cap = None
        self.is_running = False
        self.total_tracked = 0

        # Stats summary header
        header = ttk.Frame(self.root, padding=10)
        header.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(header, text="Initialize Analytics Feed", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        self.lbl_fps = ttk.Label(header, text="FPS: 0", font=("Arial", 10, "bold"))
        self.lbl_fps.pack(side=tk.LEFT, padx=15)
        self.lbl_objects = ttk.Label(header, text="Objects: 0", font=("Arial", 10, "bold"))
        self.lbl_objects.pack(side=tk.LEFT, padx=15)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn.configure(text="Halt Analytics")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn.configure(text="Initialize Analytics Feed")

    def loop(self):
        if not self.is_running:
            return

        t_start = cv2.getTickCount()
        ret, frame = self.cap.read()
        if ret:
            results = self.model.track(frame, persist=True, verbose=False)
            boxes = results[0].boxes

            obj_count = len(boxes)
            t_end = cv2.getTickCount()
            fps = int(cv2.getTickFrequency() / (t_end - t_start + 1e-5))

            self.lbl_fps.configure(text=f"FPS: {fps}")
            self.lbl_objects.configure(text=f"Active Objects: {obj_count}")

            annotated = results[0].plot()
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((900, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(15, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalyticsDashboardApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()