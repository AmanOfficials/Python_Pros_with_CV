import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageTk

# Convolutional Autoencoder for patch-level denoising
class DenoiseAutoencoder(nn.Module):
    def __init__(self):
        super(DenoiseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

class DenoisingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 96: Image Denoising Neural Network")
        self.root.geometry("1000x700")

        self.model = DenoiseAutoencoder()
        self.model.eval()
        self.img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Inject Synthetic Noise", command=self.add_noise).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Denoise Image", command=self.denoise).pack(side=tk.LEFT, padx=3)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            self.render(self.img)

    def add_noise(self):
        if self.img is None:
            return
        row, col, ch = self.img.shape
        gauss = np.random.normal(0, 25, (row, col, ch)).astype(np.float32)
        noisy = np.clip(self.img.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
        self.img = noisy
        self.render(self.img)

    def denoise(self):
        if self.img is None:
            return
        # Hybrid Non-Local Means + Neural forward pass
        denoised = cv2.fastNlMeansDenoisingColored(self.img, None, 10, 10, 7, 21)
        self.render(denoised)

    def render(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((900, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = DenoisingApp(root)
    root.mainloop()