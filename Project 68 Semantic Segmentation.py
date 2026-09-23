import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image, ImageTk

class SemanticSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 68: Semantic Segmentation (DeepLabV3)")
        self.root.geometry("1000x700")

        self.model = models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(400),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Run Semantic Segmentation", command=self.segment).pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img_path = path
            cv_img = cv2.imread(path)
            self.render(cv_img)

    def segment(self):
        if not hasattr(self, 'img_path'):
            return

        pil_img = Image.open(self.img_path).convert('RGB')
        orig_w, orig_h = pil_img.size
        tensor = self.transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            out = self.model(tensor)['out']
            seg_map = out.argmax(1).squeeze().numpy()

        # Generate color paletted overlay
        palette = np.random.randint(0, 255, (21, 3), dtype='uint8')
        color_seg = palette[seg_map]
        color_seg = cv2.resize(color_seg, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        base_bgr = cv2.imread(self.img_path)
        blended = cv2.addWeighted(base_bgr, 0.6, color_seg, 0.4, 0)
        self.render(blended)

    def render(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((900, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = SemanticSegmentationApp(root)
    root.mainloop()