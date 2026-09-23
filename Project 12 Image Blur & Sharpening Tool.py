import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class BlurSharpenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 12: Blur & Sharpening Tool")
        self.root.geometry("1000x700")

        self.cv_img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)

        ttk.Label(toolbar, text="Mode:").pack(side=tk.LEFT, padx=(10, 2))
        self.op_mode = ttk.Combobox(toolbar, values=["Unsharp Sharpen", "Custom Kernel Sharpen", "Gaussian Blur", "Box Blur"], state="readonly")
        self.op_mode.current(0)
        self.op_mode.bind("<<ComboboxSelected>>", self.apply_transformation)
        self.op_mode.pack(side=tk.LEFT)

        ttk.Label(toolbar, text="Strength:").pack(side=tk.LEFT, padx=(10, 2))
        self.slider = ttk.Scale(toolbar, from_=1, to=15, value=3, command=self.apply_transformation)
        self.slider.pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.cv_img = cv2.imread(path)
            self.apply_transformation()

    def apply_transformation(self, _=None):
        if self.cv_img is None:
            return

        strength = int(self.slider.get())
        if strength % 2 == 0:
            strength += 1

        mode = self.op_mode.get()

        if mode == "Gaussian Blur":
            processed = cv2.GaussianBlur(self.cv_img, (strength * 2 + 1, strength * 2 + 1), 0)
        elif mode == "Box Blur":
            processed = cv2.blur(self.cv_img, (strength * 2 + 1, strength * 2 + 1))
        elif mode == "Custom Kernel Sharpen":
            # 2D high-pass sharpening convolution kernel
            kernel = np.array([[0, -1, 0],
                               [-1, 4 + strength, -1],
                               [0, -1, 0]])
            processed = cv2.filter2D(self.cv_img, -1, kernel)
        elif mode == "Unsharp Sharpen":
            gaussian = cv2.GaussianBlur(self.cv_img, (0, 0), 2.0)
            processed = cv2.addWeighted(self.cv_img, 1.0 + (strength * 0.2), gaussian, -(strength * 0.2), 0)

        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((900, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = BlurSharpenApp(root)
    root.mainloop()