import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import torch
from torchvision import models, transforms
from PIL import Image, ImageTk

class WasteClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 59: Waste Classification (Recycling)")
        self.root.geometry("900x700")

        self.model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Trash Item", command=self.classify_waste).pack(side=tk.LEFT, padx=4)
        self.lbl_type = ttk.Label(toolbar, text="Category: Unknown", font=("Arial", 12, "bold"))
        self.lbl_type.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def classify_waste(self):
        path = filedialog.askopenfilename()
        if path:
            cv_img = cv2.imread(path)
            pil_img = Image.open(path).convert('RGB')
            tensor = self.transform(pil_img).unsqueeze(0)

            with torch.no_grad():
                preds = self.model(tensor)
                idx = preds.argmax().item()

            # Map ImageNet IDs to practical waste bins
            if idx in [898, 899, 907]: # Bottles, flasks
                category = "Plastic / Glass (Recyclable)"
            elif idx in [504, 600, 728]: # Cans, tins
                category = "Metal (Recyclable)"
            elif idx in [499, 815, 921]: # Cartons, paper
                category = "Paper / Cardboard"
            else:
                category = "General / Non-Recyclable Waste"

            self.lbl_type.configure(text=f"Category: {category}")

            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            preview = Image.fromarray(rgb)
            preview.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(preview)
            self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = WasteClassificationApp(root)
    root.mainloop()