import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

class InstanceSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 69: Instance Segmentation (YOLOv8-Seg)")
        self.root.geometry("1000x700")

        self.model = YOLO("yolov8n-seg.pt")
        self.img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Segment Instances", command=self.run_instance_seg).pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            self.render(self.img)

    def run_instance_seg(self):
        if self.img is None:
            return

        results = self.model.predict(self.img, verbose=False)
        annotated = results[0].plot()
        self.render(annotated)

    def render(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((900, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = InstanceSegmentationApp(root)
    root.mainloop()