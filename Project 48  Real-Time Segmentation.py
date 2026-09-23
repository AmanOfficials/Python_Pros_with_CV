import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk

class RealTimeSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 48: Real-Time Background Segmentation")
        self.root.geometry("850x650")

        self.cap = None
        self.is_running = False

        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Video Feed", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        ttk.Label(toolbar, text="Effect:").pack(side=tk.LEFT, padx=(10, 2))
        self.mode_cb = ttk.Combobox(toolbar, values=["Blur Background", "Green Screen", "Black Background"], state="readonly")
        self.mode_cb.current(0)
        self.mode_cb.pack(side=tk.LEFT)

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

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.segmenter.process(rgb)
            condition = np.stack((res.segmentation_mask,) * 3, axis=-1) > 0.5

            mode = self.mode_cb.get()
            if mode == "Blur Background":
                bg = cv2.GaussianBlur(frame, (55, 55), 0)
            elif mode == "Green Screen":
                bg = np.zeros(frame.shape, dtype=np.uint8)
                bg[:] = (0, 255, 0)
            elif mode == "Black Background":
                bg = np.zeros(frame.shape, dtype=np.uint8)

            output = np.where(condition, frame, bg)

            rgb_out = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_out)
            pil_img.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeSegmentationApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()