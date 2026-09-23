import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import torch
from torchvision import models, transforms
from PIL import Image, ImageTk

class SceneClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 66: Scene Classification")
        self.root.geometry("900x700")

        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Scene Image", command=self.classify_scene).pack(side=tk.LEFT, padx=4)
        self.lbl_scene = ttk.Label(toolbar, text="Scene: None", font=("Arial", 11, "bold"))
        self.lbl_scene.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def classify_scene(self):
        path = filedialog.askopenfilename()
        if path:
            cv_img = cv2.imread(path)
            pil_img = Image.open(path).convert('RGB')
            tensor = self.transform(pil_img).unsqueeze(0)

            with torch.no_grad():
                preds = self.model(tensor)
                idx = preds.argmax().item()

            if idx in [970, 971, 972]:
                scene_name = "Outdoor / Cliff / Valley"
            elif idx in [973, 974, 975]:
                scene_name = "Lakeside / Seashore / Coast"
            elif idx in [825, 826, 827]:
                scene_name = "Indoor / Living Room / Library"
            else:
                scene_name = f"Environment Category (Feature Index {idx})"

            self.lbl_scene.configure(text=f"Scene: {scene_name}")

            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            preview = Image.fromarray(rgb)
            preview.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(preview)
            self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = SceneClassificationApp(root)
    root.mainloop()