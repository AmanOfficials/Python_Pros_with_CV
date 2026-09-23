import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

class YOLODetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 36: YOLO Real-Time Object Detection")
        self.root.geometry("900x700")

        self.model = YOLO("yolov8n.pt")
        self.cap = None
        self.is_running = False

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start YOLO Stream", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        ttk.Label(toolbar, text="Confidence:").pack(side=tk.LEFT, padx=(10, 2))
        self.conf_slider = ttk.Scale(toolbar, from_=0.1, to=1.0, value=0.4)
        self.conf_slider.pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn.configure(text="Stop YOLO Stream")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn.configure(text="Start YOLO Stream")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            conf_val = float(self.conf_slider.get())
            results = self.model.predict(frame, conf=conf_val, verbose=False)
            annotated_frame = results[0].plot()

            rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()