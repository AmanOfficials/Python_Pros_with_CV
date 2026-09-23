import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

class ThreeDObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 90: 3D Object Bounding Box Estimation")
        self.root.geometry("900x700")

        self.model = YOLO("yolov8n.pt")
        self.cap = None
        self.is_running = False

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start 3D Projection Stream", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def draw_3d_box(self, frame, x, y, w, h):
        # Monocular pseudo-3D perspective cuboid projection
        d = int(min(w, h) * 0.3)  # Depth offset based on target dimension

        # Front face coordinates
        pt1 = (x, y)
        pt2 = (x + w, y)
        pt3 = (x + w, y + h)
        pt4 = (x, y + h)

        # Back face projected coordinates
        b1 = (x + d, y - d)
        b2 = (x + w + d, y - d)
        b3 = (x + w + d, y + h - d)
        b4 = (x + d, y + h - d)

        # Draw front face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Draw back face
        cv2.line(frame, b1, b2, (255, 100, 0), 2)
        cv2.line(frame, b2, b3, (255, 100, 0), 2)
        cv2.line(frame, b3, b4, (255, 100, 0), 2)
        cv2.line(frame, b4, b1, (255, 100, 0), 2)

        # Draw connecting depth pillars
        cv2.line(frame, pt1, b1, (0, 255, 255), 2)
        cv2.line(frame, pt2, b2, (0, 255, 255), 2)
        cv2.line(frame, pt3, b3, (0, 255, 255), 2)
        cv2.line(frame, pt4, b4, (0, 255, 255), 2)

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn.configure(text="Stop Feed")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn.configure(text="Start 3D Projection Stream")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            results = self.model.predict(frame, verbose=False)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                self.draw_3d_box(frame, x1, y1, w, h)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = ThreeDObjectDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()