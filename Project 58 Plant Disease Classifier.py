import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class PlantDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 58: Plant Disease Classifier")
        self.root.geometry("900x700")

        self.img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Leaf Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Analyze Health Status", command=self.analyze_leaf).pack(side=tk.LEFT, padx=4)

        self.lbl_status = ttk.Label(toolbar, text="Diagnosis: Ready", font=("Arial", 11, "bold"))
        self.lbl_status.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        p = filedialog.askopenfilename()
        if p:
            self.img = cv2.imread(p)
            self.render(self.img)

    def analyze_leaf(self):
        if self.img is None:
            return

        # Calculate necrotic spot coverage using HSV masking
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        # Brown/yellow necrotic lesion spectrum
        lower_brown = np.array([10, 50, 20])
        upper_brown = np.array([30, 255, 200])
        mask_disease = cv2.inRange(hsv, lower_brown, upper_brown)

        # Green leaf mask
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        green_area = cv2.countNonZero(mask_green)
        diseased_area = cv2.countNonZero(mask_disease)
        total_leaf = green_area + diseased_area

        ratio = (diseased_area / (total_leaf + 1e-5)) * 100
        if ratio > 15.0:
            diag = f"Diseased / Infected ({ratio:.1f}% necrotic area)"
            color = "red"
        else:
            diag = f"Healthy Leaf ({ratio:.1f}% spot variation)"
            color = "green"

        self.lbl_status.configure(text=f"Diagnosis: {diag}", foreground=color)

        output = self.img.copy()
        output[mask_disease > 0] = (0, 0, 255) # Highlight lesions in red
        self.render(output)

    def render(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((800, 550))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = PlantDiseaseApp(root)
    root.mainloop()