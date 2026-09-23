import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import torch
from torchvision import models, transforms
from PIL import Image, ImageTk

class AgeGroupClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 65: Age Group Classification")
        self.root.geometry("900x700")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Portrait Image", command=self.classify_age).pack(side=tk.LEFT, padx=4)
        self.lbl_age = ttk.Label(toolbar, text="Predicted Group: None", font=("Arial", 12, "bold"))
        self.lbl_age.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def classify_age(self):
        path = filedialog.askopenfilename()
        if path:
            frame = cv2.imread(path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)

            age_groups = ["Child (0-12)", "Youth (13-25)", "Adult (26-55)", "Senior (56+)"]
            
            for (x, y, w, h) in faces:
                face_crop = frame[y:y+h, x:x+w]
                pil_crop = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                tensor = self.transform(pil_crop).unsqueeze(0)

                with torch.no_grad():
                    preds = self.model(tensor)
                    group_idx = int(preds.argmax().item()) % len(age_groups)

                predicted_group = age_groups[group_idx]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, predicted_group, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.lbl_age.configure(text=f"Predicted Group: {predicted_group}")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preview = Image.fromarray(rgb)
            preview.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(preview)
            self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = AgeGroupClassifierApp(root)
    root.mainloop()