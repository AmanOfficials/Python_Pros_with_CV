import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class AccidentDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 52: Accident Detection Prototype")
        self.root.geometry("900x650")

        self.cap = None
        self.is_running = False
        self.detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Video", command=self.load_video).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Live Webcam", command=lambda: self.start_stream(0)).pack(side=tk.LEFT, padx=4)

        self.alert_lbl = ttk.Label(toolbar, text="STATUS: NORMAL", foreground="green", font=("Arial", 12, "bold"))
        self.alert_lbl.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_video(self):
        path = filedialog.askopenfilename()
        if path:
            self.start_stream(path)

    def start_stream(self, src):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(src)
        if self.cap.isOpened():
            self.is_running = True
            self.loop()

    def check_intersection(self, box1, box2):
        # Intersection-over-Union collision detection
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        return (xi2 > xi1) and (yi2 > yi1)

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            mask = self.detector.apply(frame)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for c in cnts:
                if cv2.contourArea(c) > 2500:
                    boxes.append(cv2.boundingRect(c))

            accident_alert = False
            # Check pairwise box overlaps (collision event indicator)
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if self.check_intersection(boxes[i], boxes[j]):
                        accident_alert = True
                        cv2.rectangle(frame, (boxes[i][0], boxes[i][1]), 
                                      (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), (0, 0, 255), 3)
                        cv2.rectangle(frame, (boxes[j][0], boxes[j][1]), 
                                      (boxes[j][0] + boxes[j][2], boxes[j][1] + boxes[j][3]), (0, 0, 255), 3)

            if accident_alert:
                self.alert_lbl.configure(text="STATUS: COLLISION DETECTED!", foreground="red")
                cv2.putText(frame, "CRASH / COLLISION WARNING", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                self.alert_lbl.configure(text="STATUS: NORMAL", foreground="green")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((850, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

            self.root.after(20, self.loop)
        else:
            self.is_running = False

if __name__ == "__main__":
    root = tk.Tk()
    app = AccidentDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()