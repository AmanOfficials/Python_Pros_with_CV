import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageTk
import os

class ImageForgeryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 81: Image Forgery Detection (ELA)")
        self.root.geometry("1000x700")

        self.img_path = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Run Error Level Analysis", command=self.run_ela).pack(side=tk.LEFT, padx=4)

        ttk.Label(toolbar, text="Tampering Sensitivity:").pack(side=tk.LEFT, padx=(10, 2))
        self.scale_slider = ttk.Scale(toolbar, from_=10, to=50, value=25)
        self.scale_slider.pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if path:
            self.img_path = path
            raw = Image.open(path)
            raw.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(raw)
            self.display.configure(image=self.tk_photo)

    def run_ela(self):
        if not self.img_path:
            return

        temp_resave = "temp_ela_resave.jpg"
        orig = Image.open(self.img_path).convert('RGB')
        # Re-save at a fixed 90% JPEG compression factor
        orig.save(temp_resave, 'JPEG', quality=90)
        resaved = Image.open(temp_resave)

        # Calculate absolute pixel difference
        orig_arr = np.array(orig, dtype=np.float32)
        resaved_arr = np.array(resaved, dtype=np.float32)
        diff = np.abs(orig_arr - resaved_arr)

        scale = float(self.scale_slider.get())
        ela_img = diff * scale
        ela_img = np.clip(ela_img, 0, 255).astype(np.uint8)

        # Clean temporary file
        if os.path.exists(temp_resave):
            os.remove(temp_resave)

        pil_img = Image.fromarray(ela_img)
        pil_img.thumbnail((850, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageForgeryApp(root)
    root.mainloop()