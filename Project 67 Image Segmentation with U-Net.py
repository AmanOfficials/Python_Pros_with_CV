import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import torch
import torch.nn as nn
from PIL import Image, ImageTk

# Canonical U-Net block architecture
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = UNetBlock(3, 16)
        self.pool = nn.MaxPool2d(2, 2)
        self.middle = UNetBlock(16, 32)
        self.up = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.up_conv = UNetBlock(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        m = self.middle(self.pool(d1))
        u = self.up(m)
        c = self.up_conv(torch.cat([u, d1], dim=1))
        return torch.sigmoid(self.out(c))

class UNetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 67: U-Net Image Segmentation")
        self.root.geometry("950x700")

        self.model = UNet()
        self.model.eval()
        self.img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Segment Mask", command=self.run_segmentation).pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            self.render(self.img)

    def run_segmentation(self):
        if self.img is None:
            return

        resized = cv2.resize(self.img, (256, 256))
        tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            mask = self.model(tensor).squeeze().numpy()

        binary_mask = (mask > 0.5).astype('uint8') * 255
        color_mask = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(resized, 0.7, color_mask, 0.3, 0)
        self.render(blended)

    def render(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((850, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = UNetApp(root)
    root.mainloop()