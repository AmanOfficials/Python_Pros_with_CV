import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image, ImageTk

class RealTimeSemanticApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 88: Real-Time Semantic Segmentation")
        self.root.geometry("850x650")

        self.model = models.segmentation.lraspp_mobilenet_v3_large(
            weights=models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.model.eval()

        self.cap = None
        self.is_running = False

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Feed", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

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
            self.btn.configure(text="Start Feed")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            h, w, _ = frame.shape
            low_res = cv2.resize(frame, (256, 256))
            tensor = self.preprocess(low_res).unsqueeze(0)

            with torch.no_grad():
                out = self.model(tensor)['out']
                seg = out.argmax(1).squeeze().byte().cpu().numpy()

            # Map mask to colors and upscale back to viewport
            color_mask = cv2.applyColorMap(seg * 12, cv2.COLORMAP_RAINBOW)
            color_mask = cv2.resize(color_mask, (w, h))

            blended = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)

            rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(30, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeSemanticApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()