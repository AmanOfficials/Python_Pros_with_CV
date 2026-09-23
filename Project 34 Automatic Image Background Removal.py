import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class BackgroundRemoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 34: Background Removal")
        self.root.geometry("950x700")

        self.img = None
        self.output_img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Remove Background", command=self.remove_bg).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Save Transparent PNG", command=self.save_result).pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            self.render(self.img)

    def remove_bg(self):
        if self.img is None:
            return

        h, w = self.img.shape[:2]
        mask = np.zeros(self.img.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Margin crop for GrabCut initialisation
        rect = (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90))
        cv2.grabCut(self.img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        foreground = self.img * mask2[:, :, np.newaxis]

        # Generate 4-channel BGRA (transparent alpha background)
        b, g, r = cv2.split(foreground)
        alpha = (mask2 * 255).astype(np.uint8)
        self.output_img = cv2.merge([b, g, r, alpha])

        self.render(self.output_img)

    def render(self, img):
        if img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((850, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

    def save_result(self):
        if self.output_img is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                cv2.imwrite(path, self.output_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = BackgroundRemoverApp(root)
    root.mainloop()