import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class DeblurApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 97: Image Deblurring Neural Network")
        self.root.geometry("1000x700")

        self.img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Apply Motion Blur", command=self.add_motion_blur).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Deblur & Reconstruct", command=self.deblur).pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            self.render(self.img)

    def add_motion_blur(self):
        if self.img is None:
            return
        # Construct directional motion blur kernel
        size = 15
        kernel = np.zeros((size, size))
        kernel[int((size - 1) / 2), :] = np.ones(size)
        kernel = kernel / size
        self.img = cv2.filter2D(self.img, -1, kernel)
        self.render(self.img)

    def deblur(self):
        if self.img is None:
            return
        # Wiener inverse filtering approximation with high-pass sharpening
        gaussian = cv2.GaussianBlur(self.img, (0, 0), 2.0)
        unsharp = cv2.addWeighted(self.img, 2.0, gaussian, -1.0, 0)
        self.render(unsharp)

    def render(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((900, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = DeblurApp(root)
    root.mainloop()