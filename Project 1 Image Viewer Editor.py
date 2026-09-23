import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ImageViewerEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 1: Image Viewer & Editor")
        self.root.geometry("1100x750")

        # Stores the original image and current edited state
        self.original_img = None
        self.edited_img = None

        # Build Sidebar Controls and Canvas
        self._build_sidebar()
        self._build_viewport()

    def _build_sidebar(self):
        sidebar = ttk.Frame(self.root, padding=10, width=280)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(sidebar, text="Open Image", command=self.load_image).pack(fill=tk.X, pady=4)
        ttk.Button(sidebar, text="Save Output", command=self.save_image).pack(fill=tk.X, pady=4)
        ttk.Button(sidebar, text="Reset Original", command=self.reset_image).pack(fill=tk.X, pady=4)

        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Brightness Slider (-100 to 100)
        ttk.Label(sidebar, text="Brightness").pack(anchor=tk.W)
        self.brightness_slider = ttk.Scale(sidebar, from_=-100, to=100, value=0, command=self.apply_transforms)
        self.brightness_slider.pack(fill=tk.X, pady=2)

        # Contrast Slider (0.5 to 3.0)
        ttk.Label(sidebar, text="Contrast").pack(anchor=tk.W)
        self.contrast_slider = ttk.Scale(sidebar, from_=0.5, to=3.0, value=1.0, command=self.apply_transforms)
        self.contrast_slider.pack(fill=tk.X, pady=2)

        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Rotation and Flipping buttons
        ttk.Button(sidebar, text="Rotate 90° Clockwise", command=self.rotate_clockwise).pack(fill=tk.X, pady=3)
        ttk.Button(sidebar, text="Flip Horizontally", command=self.flip_horizontal).pack(fill=tk.X, pady=3)
        ttk.Button(sidebar, text="Flip Vertically", command=self.flip_vertical).pack(fill=tk.X, pady=3)

    def _build_viewport(self):
        self.canvas = tk.Canvas(self.root, bg="#2b2b2b")
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.original_img = cv2.imread(path)
            self.reset_image()

    def reset_image(self):
        if self.original_img is not None:
            self.edited_img = self.original_img.copy()
            self.brightness_slider.set(0)
            self.contrast_slider.set(1.0)
            self.render_image(self.edited_img)

    def apply_transforms(self, _=None):
        if self.original_img is None:
            return
        # Formula: new_img = alpha * original_img + beta
        alpha = float(self.contrast_slider.get())
        beta = float(self.brightness_slider.get())
        adjusted = cv2.convertScaleAbs(self.edited_img, alpha=alpha, beta=beta)
        self.render_image(adjusted)

    def rotate_clockwise(self):
        if self.edited_img is not None:
            self.edited_img = cv2.rotate(self.edited_img, cv2.ROTATE_90_CLOCKWISE)
            self.apply_transforms()

    def flip_horizontal(self):
        if self.edited_img is not None:
            self.edited_img = cv2.flip(self.edited_img, 1)
            self.apply_transforms()

    def flip_vertical(self):
        if self.edited_img is not None:
            self.edited_img = cv2.flip(self.edited_img, 0)
            self.apply_transforms()

    def render_image(self, img_bgr):
        # Convert OpenCV BGR format to RGB for Tkinter compatibility
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Scale down dynamically to fit canvas size
        canvas_width = max(self.canvas.winfo_width(), 400)
        canvas_height = max(self.canvas.winfo_height(), 400)
        pil_img.thumbnail((canvas_width, canvas_height))

        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=self.tk_img)

    def save_image(self):
        if self.edited_img is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".png")
            if save_path:
                cv2.imwrite(save_path, self.edited_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewerEditorApp(root)
    root.mainloop()