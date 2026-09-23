import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class VehicleCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 21: Vehicle Counter")
        self.root.geometry("900x700")

        self.cap = None
        self.is_running = False
        self.count = 0
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50, detectShadows=False)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Video File", command=self.load_video).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Use Webcam", command=self.use_webcam).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Reset Count", command=self.reset_count).pack(side=tk.LEFT, padx=4)

        self.counter_lbl = ttk.Label(toolbar, text="Vehicles Detected: 0", font=("Arial", 12, "bold"))
        self.counter_lbl.pack(side=tk.LEFT, padx=20)

        self.viewport = ttk.Label(self.root)
        self.viewport.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if path:
            self.start_stream(path)

    def use_webcam(self):
        self.start_stream(0)

    def start_stream(self, source):
        self.stop_stream()
        self.cap = cv2.VideoCapture(source)
        if self.cap.isOpened():
            self.is_running = True
            self.stream_loop()

    def stop_stream(self):
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def reset_count(self):
        self.count = 0
        self.counter_lbl.configure(text="Vehicles Detected: 0")

    def stream_loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            h, w, _ = frame.shape
            line_y = int(h * 0.6)

            # Draw virtual detection line
            cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)

            # Background subtraction & noise removal
            mask = self.subtractor.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < 1800:
                    continue

                (x, y, cw, ch) = cv2.boundingRect(c)
                cy = int(y + ch / 2)
                cx = int(x + cw / 2)

                cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

                # Count vehicle if centroid intersects line zone
                if line_y - 6 <= cy <= line_y + 6:
                    self.count += 1
                    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 0), 3)

            self.counter_lbl.configure(text=f"Vehicles Detected: {self.count}")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.viewport.configure(image=self.tk_photo)

            self.root.after(20, self.stream_loop)
        else:
            self.stop_stream()

if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_stream(), root.destroy()))
    root.mainloop()