import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class PeopleAnalyticsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 44: People Counting & Footfall Analytics")
        self.root.geometry("900x700")

        self.cap = None
        self.is_running = False
        self.detector = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50)
        self.count = 0
        self.history = [0] * 20

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Video Feed", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        self.stats = ttk.Label(toolbar, text="Occupancy: 0 | Peak Today: 0", font=("Arial", 11, "bold"))
        self.stats.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn.configure(text="Stop Video Feed")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn.configure(text="Start Video Feed")

    def draw_analytics_overlay(self, frame, current_count):
        self.history.append(current_count)
        self.history.pop(0)

        # Draw mini-chart bar overlay in bottom-left corner
        h, _, _ = frame.shape
        cv2.rectangle(frame, (10, h - 90), (220, h - 10), (30, 30, 30), -1)
        for i, val in enumerate(self.history):
            bar_height = min(int(val * 12), 70)
            cv2.line(frame, (20 + i * 10, h - 15), (20 + i * 10, h - 15 - bar_height), (0, 255, 255), 3)

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            mask = self.detector.apply(frame)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            visible_count = 0
            for c in cnts:
                if cv2.contourArea(c) > 2000:
                    visible_count += 1
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self.stats.configure(text=f"Current Occupancy: {visible_count}")
            self.draw_analytics_overlay(frame, visible_count)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(30, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = PeopleAnalyticsApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()