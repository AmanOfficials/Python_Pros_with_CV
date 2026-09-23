import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ColorDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 6: HSV Color Detection System")
        self.root.geometry("1100x750")

        self.img = None

        controls = ttk.Frame(self.root, padding=10)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(controls, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=4)

        # HSV Range Sliders
        self.sliders = {}
        ranges = [("Lower H", 0, 179, 0), ("Upper H", 0, 179, 179),
                  ("Lower S", 0, 255, 50), ("Upper S", 0, 255, 255),
                  ("Lower V", 0, 255, 50), ("Upper V", 0, 255, 255)]

        for name, min_val, max_val, default in ranges:
            ttk.Label(controls, text=name).pack(anchor=tk.W, pady=(4, 0))
            s = ttk.Scale(controls, from_=min_val, to=max_val, value=default, command=self.update_mask)
            s.pack(fill=tk.X)
            self.sliders[name] = s

        self.canvas = ttk.Label(self.root)
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            self.update_mask()

    def update_mask(self, _=None):
        if self.img is None:
            return

        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower = np.array([int(self.sliders["Lower H"].get()),
                          int(self.sliders["Lower S"].get()),
                          int(self.sliders["Lower V"].get())])
        upper = np.array([int(self.sliders["Upper H"].get()),
                          int(self.sliders["Upper S"].get()),
                          int(self.sliders["Upper V"].get())])

        mask = cv2.inRange(hsv, lower, upper)
        # Keep only the matching color areas
        result = cv2.bitwise_and(self.img, self.img, mask=mask)

        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((800, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.canvas.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorDetectionApp(root)
    root.mainloop()