import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

class SmartTrafficCamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 39: Smart Traffic Camera")
        self.root.geometry("950x700")

        self.model = YOLO("yolov8n.pt")
        # COCO indices for vehicles: 2=car, 3=motorcycle, 5=bus, 7=truck
        self.vehicle_classes = [2, 3, 5, 7]

        self.cap = None
        self.is_running = False

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Video", command=self.load_video).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Live Webcam", command=lambda: self.start_stream(0)).pack(side=tk.LEFT, padx=4)

        self.stats = ttk.Label(toolbar, text="Total Vehicles Visible: 0", font=("Arial", 11, "bold"))
        self.stats.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_video(self):
        path = filedialog.askopenfilename()
        if path:
            self.start_stream(path)

    def start_stream(self, src):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(src)
        if self.cap.isOpened():
            self.is_running = True
            self.loop()

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            results = self.model.predict(frame, classes=self.vehicle_classes, verbose=False)
            count = len(results[0].boxes)
            self.stats.configure(text=f"Total Vehicles Visible: {count}")
            annotated = results[0].plot()

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

            self.root.after(20, self.loop)
        else:
            self.is_running = False

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartTrafficCamApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()