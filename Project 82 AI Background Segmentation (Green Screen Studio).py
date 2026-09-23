import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk

class GreenScreenStudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 82: AI Background Segmentation Studio")
        self.root.geometry("900x700")

        self.cap = None
        self.is_running = False
        self.bg_img = None

        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn_cam = ttk.Button(toolbar, text="Start Feed", command=self.toggle_stream)
        self.btn_cam.pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Load Custom Background", command=self.load_custom_bg).pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_custom_bg(self):
        path = filedialog.askopenfilename()
        if path:
            self.bg_img = cv2.imread(path)

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn_cam.configure(text="Stop Feed")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn_cam.configure(text="Start Feed")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.segmenter.process(rgb)

            # Binary silhouette condition
            condition = np.stack((res.segmentation_mask,) * 3, axis=-1) > 0.55

            if self.bg_img is not None:
                bg = cv2.resize(self.bg_img, (w, h))
            else:
                bg = np.zeros(frame.shape, dtype=np.uint8)
                bg[:] = (0, 180, 0)  # Standard backdrop green

            output = np.where(condition, frame, bg)

            rgb_out = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_out)
            pil_img.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = GreenScreenStudioApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()