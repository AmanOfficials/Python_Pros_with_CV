import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import torch
from torchvision import models, transforms
from PIL import Image, ImageTk

class FashionClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 61: Fashion Image Classifier")
        self.root.geometry("900x700")

        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Apparel Image", command=self.classify_fashion).pack(side=tk.LEFT, padx=4)
        self.res_lbl = ttk.Label(toolbar, text="Apparel Category: None", font=("Arial", 11, "bold"))
        self.res_lbl.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def classify_fashion(self):
        path = filedialog.askopenfilename()
        if path:
            cv_img = cv2.imread(path)
            pil_img = Image.open(path).convert('RGB')
            tensor = self.transform(pil_img).unsqueeze(0)

            with torch.no_grad():
                preds = self.model(tensor)
                idx = preds.argmax().item()

            # ImageNet fashion and wearable indices
            if idx in [414, 834, 842]:
                category = "Suit / Blazer / Tie"
            elif idx in [515, 617, 852]:
                category = "Shoes / Loafers / Boots"
            elif idx in [611, 840, 879]:
                category = "Umbrella / Hat / Cap"
            elif idx in [600, 655, 770]:
                category = "Bag / Backpack / Handbag"
            else:
                category = f"Apparel Accessory / Fabric (Class {idx})"

            self.res_lbl.configure(text=f"Apparel Category: {category}")

            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            preview = Image.fromarray(rgb)
            preview.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(preview)
            self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = FashionClassifierApp(root)
    root.mainloop()