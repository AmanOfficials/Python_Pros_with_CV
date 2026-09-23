import tkinter as tk
from tkinter import ttk
from collections import deque
import cv2
import numpy as np
from PIL import Image, ImageTk

class WebcamColorTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 14: Webcam Color Tracker")
        self.root.geometry("850x650")

        self.cap = None
        self.is_running = False
        self.pts = deque(maxlen=32)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.toggle_btn = ttk.Button(toolbar, text="Start Stream", command=self.toggle_camera)
        self.toggle_btn.pack(side=tk.LEFT, padx=4)

        ttk.Label(toolbar, text="Tracking Color:").pack(side=tk.LEFT, padx=(10, 2))
        self.color_choice = ttk.Combobox(toolbar, values=["Green", "Blue", "Red"], state="readonly")
        self.color_choice.current(0)
        self.color_choice.pack(side=tk.LEFT)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.toggle_btn.configure(text="Stop Stream")
                self.stream_loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.toggle_btn.configure(text="Start Stream")

    def get_color_ranges(self, name):
        if name == "Green":
            return np.array([35, 80, 80]), np.array([85, 255, 255])
        elif name == "Blue":
            return np.array([95, 100, 100]), np.array([130, 255, 255])
        elif name == "Red":
            return np.array([0, 120, 100]), np.array([10, 255, 255])

    def stream_loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1) # Mirror preview
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower, upper = self.get_color_ranges(self.color_choice.get())

            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            center = None

            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                if radius > 10:
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

            self.pts.appendleft(center)

            # Draw trajectory path line
            for i in range(1, len(self.pts)):
                if self.pts[i - 1] is None or self.pts[i] is None:
                    continue
                thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
                cv2.line(frame, self.pts[i - 1], self.pts[i], (0, 255, 0), thickness)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(20, self.stream_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamColorTrackerApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()