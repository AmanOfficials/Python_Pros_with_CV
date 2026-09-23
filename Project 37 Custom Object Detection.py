import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

class CustomDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 37: Custom Object Detection")
        self.root.geometry("950x700")

        self.model = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Custom Model (.pt)", command=self.load_model).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Inference on Image", command=self.run_inference).pack(side=tk.LEFT, padx=4)

        self.status = ttk.Label(toolbar, text="No model loaded", font=("Arial", 10, "italic"))
        self.status.pack(side=tk.LEFT, padx=10)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("YOLO Model Weights", "*.pt *.onnx")])
        if path:
            self.model = YOLO(path)
            self.status.configure(text=f"Loaded: {path.split('/')[-1]}")

    def run_inference(self):
        if self.model is None:
            self.status.configure(text="Please load a model file first!")
            return

        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if path:
            results = self.model.predict(path)
            annotated = results[0].plot()

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = CustomDetectorApp(root)
    root.mainloop()