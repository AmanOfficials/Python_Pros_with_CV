import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import torch
from torchvision import models, transforms
from PIL import Image, ImageTk
import json
import urllib.request

class CNNClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 56: CNN Image Classifier")
        self.root.geometry("900x700")

        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.model.eval()

        # Download ImageNet class labels
        labels_url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
        try:
            with urllib.request.urlopen(labels_url) as url:
                self.labels = {int(k): v[1] for k, v in json.loads(url.read().decode()).items()}
        except Exception:
            self.labels = {i: f"Class_{i}" for i in range(1000)}

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image & Classify", command=self.classify_image).pack(side=tk.LEFT, padx=4)
        self.pred_lbl = ttk.Label(toolbar, text="Prediction: Ready", font=("Arial", 11, "bold"))
        self.pred_lbl.pack(side=tk.LEFT, padx=15)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def classify_image(self):
        path = filedialog.askopenfilename()
        if path:
            cv_img = cv2.imread(path)
            pil_img = Image.open(path).convert('RGB')
            tensor = self.preprocess(pil_img).unsqueeze(0)

            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                top_prob, top_catid = torch.topk(probs, 1)

            cat_name = self.labels.get(top_catid[0].item(), "Unknown")
            self.pred_lbl.configure(text=f"Prediction: {cat_name} ({top_prob[0].item()*100:.1f}%)")

            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            preview = Image.fromarray(rgb)
            preview.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(preview)
            self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = CNNClassifierApp(root)
    root.mainloop()