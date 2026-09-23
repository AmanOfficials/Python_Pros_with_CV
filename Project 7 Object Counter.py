import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ObjectCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 7: Object Counter")
        self.root.geometry("1000x700")

        self.img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)

        ttk.Label(toolbar, text="Min Area Filter:").pack(side=tk.LEFT, padx=(15, 2))
        self.area_slider = ttk.Scale(toolbar, from_=50, to=5000, value=300, command=self.count_objects)
        self.area_slider.pack(side=tk.LEFT, padx=5)

        self.count_label = ttk.Label(toolbar, text="Detected Objects: 0", font=("Arial", 11, "bold"))
        self.count_label.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            self.count_objects()

    def count_objects(self, _=None):
        if self.img is None:
            return

        min_area = self.area_slider.get()
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Convert to binary
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output = self.img.copy()
        count = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area > min_area:
                count += 1
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(output, f"#{count}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        self.count_label.configure(text=f"Detected Objects: {count}")

        rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((900, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectCounterApp(root)
    root.mainloop()