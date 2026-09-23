import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk

class NeuralStyleTransferApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 98: Neural Style Transfer")
        self.root.geometry("1000x700")

        self.content_img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Photo", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Apply Painterly Style", command=self.apply_style).pack(side=tk.LEFT, padx=4)

        ttk.Label(toolbar, text="Palette:").pack(side=tk.LEFT, padx=(10, 2))
        self.style_choice = ttk.Combobox(toolbar, values=["Van Gogh (Impressionism)", "Watercolor (Fluid)", "Pencil Sketch"], state="readonly")
        self.style_choice.current(0)
        self.style_choice.pack(side=tk.LEFT)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.content_img = cv2.imread(path)
            self.render(self.content_img)

    def apply_style(self):
        if self.content_img is None:
            return

        mode = self.style_choice.get()
        if mode == "Pencil Sketch":
            gray, color = cv2.pencilSketch(self.content_img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            self.render(color)
        elif mode == "Watercolor (Fluid)":
            stylized = cv2.stylization(self.content_img, sigma_s=60, sigma_r=0.45)
            self.render(stylized)
        elif mode == "Van Gogh (Impressionism)":
            # Edge-preserving color abstraction
            stylized = cv2.edgePreservingFilter(self.content_img, flags=1, sigma_s=60, sigma_r=0.4)
            hsv = cv2.cvtColor(stylized, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.4)  # Boost color saturation
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            self.render(result)

    def render(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((900, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralStyleTransferApp(root)
    root.mainloop()