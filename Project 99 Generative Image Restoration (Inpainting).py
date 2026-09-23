import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ImageInpaintingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 99: Generative Image Restoration (Inpainting)")
        self.root.geometry("1000x750")

        self.img = None
        self.mask = None
        self.h, self.w = 0, 0

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Inpaint & Restore", command=self.inpaint).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Reset Mask", command=self.reset_mask).pack(side=tk.LEFT, padx=4)

        self.canvas = tk.Canvas(self.root, bg="#202020")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint_brush)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            self.h, self.w, _ = self.img.shape
            self.reset_mask()
            self.render()

    def reset_mask(self):
        if self.img is not None:
            self.mask = np.zeros((self.h, self.w), dtype=np.uint8)
            self.render()

    def paint_brush(self, event):
        if self.mask is not None:
            r = 8
            # Mark mask region
            cv2.circle(self.mask, (event.x, event.y), r, 255, -1)
            # Draw brush stroke on canvas
            self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="red", outline="red")

    def inpaint(self):
        if self.img is None or self.mask is None:
            return
        # Navier-Stokes / Telea algorithmic inpainting
        restored = cv2.inpaint(self.img, self.mask, 3, cv2.INPAINT_TELEA)
        self.img = restored
        self.mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self.render()

    def render(self):
        rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.pil_img = Image.fromarray(rgb)
        self.tk_photo = ImageTk.PhotoImage(self.pil_img)
        self.canvas.config(width=self.w, height=self.h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageInpaintingApp(root)
    root.mainloop()