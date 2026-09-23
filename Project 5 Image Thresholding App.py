import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk

class ThresholdingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 5: Image Thresholding App")
        self.root.geometry("1000x700")

        self.img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=3)

        ttk.Label(toolbar, text="Threshold Type:").pack(side=tk.LEFT, padx=(10, 2))
        self.method_cb = ttk.Combobox(toolbar, values=[
            "Binary", "Binary Invert", "Truncate", "ToZero", "Otsu Threshold", "Adaptive Gaussian"
        ], state="readonly")
        self.method_cb.current(0)
        self.method_cb.bind("<<ComboboxSelected>>", self.apply_threshold)
        self.method_cb.pack(side=tk.LEFT, padx=3)

        ttk.Label(toolbar, text="Threshold Value:").pack(side=tk.LEFT, padx=(10, 2))
        self.thresh_slider = ttk.Scale(toolbar, from_=0, to=255, value=127, command=self.apply_threshold)
        self.thresh_slider.pack(side=tk.LEFT)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.apply_threshold()

    def apply_threshold(self, _=None):
        if self.img is None:
            return

        t_val = int(self.thresh_slider.get())
        method = self.method_cb.get()

        if method == "Binary":
            _, res = cv2.threshold(self.img, t_val, 255, cv2.THRESH_BINARY)
        elif method == "Binary Invert":
            _, res = cv2.threshold(self.img, t_val, 255, cv2.THRESH_BINARY_INV)
        elif method == "Truncate":
            _, res = cv2.threshold(self.img, t_val, 255, cv2.THRESH_TRUNC)
        elif method == "ToZero":
            _, res = cv2.threshold(self.img, t_val, 255, cv2.THRESH_TOZERO)
        elif method == "Otsu Threshold":
            # Otsu calculates the optimal threshold automatically
            _, res = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "Adaptive Gaussian":
            res = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        pil_img = Image.fromarray(res)
        pil_img.thumbnail((900, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = ThresholdingApp(root)
    root.mainloop()