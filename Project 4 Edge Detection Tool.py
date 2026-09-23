import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class EdgeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 4: Edge Detection Tool")
        self.root.geometry("1000x700")

        self.original_img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=3)

        ttk.Label(toolbar, text="Method:").pack(side=tk.LEFT, padx=(10, 2))
        self.method_cb = ttk.Combobox(toolbar, values=["Canny", "Sobel X", "Sobel Y", "Sobel Combined", "Laplacian"], state="readonly")
        self.method_cb.current(0)
        self.method_cb.bind("<<ComboboxSelected>>", self.detect_edges)
        self.method_cb.pack(side=tk.LEFT, padx=3)

        ttk.Label(toolbar, text="Threshold 1:").pack(side=tk.LEFT, padx=(10, 2))
        self.t1_slider = ttk.Scale(toolbar, from_=0, to=255, value=100, command=self.detect_edges)
        self.t1_slider.pack(side=tk.LEFT)

        ttk.Label(toolbar, text="Threshold 2:").pack(side=tk.LEFT, padx=(10, 2))
        self.t2_slider = ttk.Scale(toolbar, from_=0, to=255, value=200, command=self.detect_edges)
        self.t2_slider.pack(side=tk.LEFT)

        self.canvas_label = ttk.Label(self.root)
        self.canvas_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.original_img = cv2.imread(path)
            self.detect_edges()

    def detect_edges(self, _=None):
        if self.original_img is None:
            return

        gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        method = self.method_cb.get()
        t1 = int(self.t1_slider.get())
        t2 = int(self.t2_slider.get())

        if method == "Canny":
            output = cv2.Canny(gray, t1, t2)
        elif method == "Sobel X":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            output = cv2.convertScaleAbs(sobelx)
        elif method == "Sobel Y":
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            output = cv2.convertScaleAbs(sobely)
        elif method == "Sobel Combined":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            output = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))
        elif method == "Laplacian":
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            output = cv2.convertScaleAbs(lap)

        pil_img = Image.fromarray(output)
        pil_img.thumbnail((900, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.canvas_label.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = EdgeDetectionApp(root)
    root.mainloop()