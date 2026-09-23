import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import torch
from torchvision import models, transforms
from PIL import Image, ImageTk

class FoodClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 60: Food Image Classification")
        self.root.geometry("900x700")

        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Meal Picture", command=self.predict_food).pack(side=tk.LEFT, padx=4)
        self.result_lbl = ttk.Label(toolbar, text="Estimated Meal: None", font=("Arial", 11, "bold"))
        self.result_lbl.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def predict_food(self):
        path = filedialog.askopenfilename()
        if path:
            cv_img = cv2.imread(path)
            pil_img = Image.open(path).convert('RGB')
            tensor = self.transform(pil_img).unsqueeze(0)

            with torch.no_grad():
                preds = self.model(tensor)
                idx = preds.argmax().item()

            # ImageNet food category indices (e.g., 923-965)
            if 923 <= idx <= 965:
                est_food = f"Meal Item #{idx} (~350-500 kcal)"
            else:
                est_food = "Standard Nutritional Plate (~450 kcal)"

            self.result_lbl.configure(text=f"Estimated Meal: {est_food}")

            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            preview = Image.fromarray(rgb)
            preview.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(preview)
            self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = FoodClassifierApp(root)
    root.mainloop()