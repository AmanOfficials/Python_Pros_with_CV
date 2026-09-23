import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ShapeDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 8: Shape Detection")
        self.root.geometry("1000x700")

        self.img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            self.detect_shapes()

    def detect_shapes(self):
        if self.img is None:
            return

        output = self.img.copy()
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 200:
                continue

            # Approximate polygonal curves
            epsilon = 0.04 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            corners = len(approx)

            x, y, w, h = cv2.boundingRect(approx)
            shape_name = "Polygon"

            if corners == 3:
                shape_name = "Triangle"
            elif corners == 4:
                aspect_ratio = float(w) / h
                shape_name = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
            elif corners > 4:
                shape_name = "Circle"

            cv2.drawContours(output, [approx], 0, (255, 0, 0), 2)
            cv2.putText(output, shape_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((900, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = ShapeDetectorApp(root)
    root.mainloop()