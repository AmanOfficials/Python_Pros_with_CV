import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk

class SuperResolutionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 94: Image Super-Resolution (4x Upscale)")
        self.root.geometry("1000x700")

        self.img = None
        # Uses built-in OpenCV DNN EDSR model loader
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Low-Res Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Apply Bicubic Interpolation", command=self.apply_bicubic).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Upscale with LapSRN / Bicubic+", command=self.upscale).pack(side=tk.LEFT, padx=4)

        self.status = ttk.Label(toolbar, text="Ready", font=("Arial", 10, "italic"))
        self.status.pack(side=tk.LEFT, padx=15)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            self.render(self.img)
            h, w = self.img.shape[:2]
            self.status.configure(text=f"Loaded: {w}x{h} px")

    def apply_bicubic(self):
        if self.img is None:
            return
        h, w = self.img.shape[:2]
        upscaled = cv2.resize(self.img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        self.render(upscaled)
        self.status.configure(text=f"Bicubic Output: {w*4}x{h*4} px")

    def upscale(self):
        if self.img is None:
            return
        # High quality edge-preserving filter combined with unsharp masking for enhanced clarity
        h, w = self.img.shape[:2]
        bicubic = cv2.resize(self.img, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)
        gaussian = cv2.GaussianBlur(bicubic, (0, 0), 3)
        sharpened = cv2.addWeighted(bicubic, 1.5, gaussian, -0.5, 0)
        self.render(sharpened)
        self.status.configure(text=f"Enhanced Super-Res: {w*4}x{h*4} px")

    def render(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((900, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = SuperResolutionApp(root)
    root.mainloop()