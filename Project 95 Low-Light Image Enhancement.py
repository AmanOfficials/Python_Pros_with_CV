import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class LowLightEnhanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 95: Low-Light Image Enhancement (Retinex)")
        self.root.geometry("1000x700")

        self.img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Low-Light Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Enhance Luminance", command=self.enhance_image).pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            self.render(self.img)

    def retinex_enhance(self, img):
        # Convert to LAB color space to separate lightness from chroma
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on L-channel
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge enhanced L-channel with original color coordinates
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Gamma correction to balance highlights
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(enhanced, table)

    def enhance_image(self):
        if self.img is None:
            return
        result = self.retinex_enhance(self.img)
        self.render(result)

    def render(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((900, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = LowLightEnhanceApp(root)
    root.mainloop()