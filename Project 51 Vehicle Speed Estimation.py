import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class SpeedEstimationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 51: Vehicle Speed Estimation")
        self.root.geometry("950x700")

        self.cap = None
        self.is_running = False
        self.ppm = 8.8  # Pixels per meter calibration factor
        self.fps = 30.0  # Assumed or extracted frame rate
        self.prev_tracks = {}  # {id: (y_pos, frame_idx)}

        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Video", command=self.load_video).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Webcam", command=lambda: self.start_stream(0)).pack(side=tk.LEFT, padx=4)

        self.status = ttk.Label(toolbar, text="Speed Monitor Active", font=("Arial", 11, "bold"))
        self.status.pack(side=tk.LEFT, padx=15)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_video(self):
        path = filedialog.askopenfilename()
        if path:
            self.start_stream(path)

    def start_stream(self, src):
        self.stop_stream()
        self.cap = cv2.VideoCapture(src)
        if self.cap.isOpened():
            fps_val = self.cap.get(cv2.CAP_PROP_FPS)
            if fps_val > 0:
                self.fps = fps_val
            self.is_running = True
            self.loop()

    def stop_stream(self):
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            mask = self.subtractor.apply(frame)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for i, c in enumerate(cnts):
                if cv2.contourArea(c) < 1800:
                    continue

                x, y, w, h = cv2.boundingRect(c)
                cy = y + h // 2
                obj_id = f"V_{x // 50}_{i}"  # Spatial bucket hash

                speed_kmh = 0
                if obj_id in self.prev_tracks:
                    prev_y, prev_frame = self.prev_tracks[obj_id]
                    d_pixels = abs(cy - prev_y)
                    d_frames = frame_idx - prev_frame
                    if d_frames > 0:
                        # Speed formula: (pixels / ppm) / (seconds) * 3.6 km/h
                        d_meters = d_pixels / self.ppm
                        d_seconds = d_frames / self.fps
                        speed_kmh = (d_meters / d_seconds) * 3.6

                self.prev_tracks[obj_id] = (cy, frame_idx)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if speed_kmh > 5:
                    cv2.putText(frame, f"{int(speed_kmh)} km/h", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

            self.root.after(20, self.loop)
        else:
            self.stop_stream()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeedEstimationApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_stream(), root.destroy()))
    root.mainloop()