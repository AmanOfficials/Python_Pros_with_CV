import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import torch
from torchvision import models, transforms
from PIL import Image, ImageTk

class AnimalClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 57: Animal Image Classifier")
        self.root.geometry("900x700")

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Animal Image", command=self.classify).pack(side=tk.LEFT, padx=4)
        self.res_lbl = ttk.Label(toolbar, text="Result: None", font=("Arial", 12, "bold"))
        self.res_lbl.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def classify(self):
        path = filedialog.askopenfilename()
        if path:
            cv_img = cv2.imread(path)
            pil_img = Image.open(path).convert('RGB')
            tensor = self.preprocess(pil_img).unsqueeze(0)

            with torch.no_grad():
                preds = self.model(tensor)
                cls_idx = preds.argmax().item()

            # ImageNet indices 151-280 map to domestic animals and dogs
            is_animal = 150 <= cls_idx <= 397
            label = f"Animal Detected (Class Index: {cls_idx})" if is_animal else f"Non-Animal or Generic Category ({cls_idx})"
            self.res_lbl.configure(text=f"Result: {label}")

            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            preview = Image.fromarray(rgb)
            preview.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(preview)
            self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = AnimalClassifierApp(root)
    root.mainloop()