import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ImageQualityAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 80: AI Image Quality Analyzer")
        self.root.geometry("1000x700")

        self.img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Analyze Quality Metrics", command=self.analyze).pack(side=tk.LEFT, padx=4)

        # Report metrics panel
        self.metrics_lbl = ttk.Label(toolbar, text="Sharpness: - | Brightness: - | Quality: -", font=("Arial", 11, "bold"))
        self.metrics_lbl.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            self.render(self.img)

    def analyze(self):
        if self.img is None:
            return

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        # Metric 1: Blur Detection via Laplacian variance
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_status = "Blurry" if sharpness < 100 else "Sharp"

        # Metric 2: Luminance/Exposure evaluation
        mean_brightness = np.mean(gray)
        if mean_brightness < 40:
            exp_status = "Underexposed"
        elif mean_brightness > 210:
            exp_status = "Overexposed"
        else:
            exp_status = "Good Exposure"

        overall = "POOR" if (sharpness < 100 or mean_brightness < 40 or mean_brightness > 210) else "EXCELLENT"
        self.metrics_lbl.configure(text=f"Sharpness: {int(sharpness)} ({blur_status}) | Exp: {int(mean_brightness)} ({exp_status}) | Status: {overall}")

    def render(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((850, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageQualityAnalyzerApp(root)
    root.mainloop()