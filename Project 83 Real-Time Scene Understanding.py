import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

class SceneUnderstandingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 83: Real-Time Scene Understanding")
        self.root.geometry("1000x700")

        self.model = YOLO("yolov8n.pt")
        self.cap = None
        self.is_running = False

        # Sidebar summary
        self.sidebar = ttk.Frame(self.root, padding=10, width=280)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(self.sidebar, text="Scene Metadata", font=("Arial", 12, "bold")).pack(pady=5)
        self.stats_text = tk.Text(self.sidebar, width=30, height=20)
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Video Understanding", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn.configure(text="Stop Video")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn.configure(text="Start Video Understanding")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            results = self.model.predict(frame, verbose=False)
            class_counts = {}

            for box in results[0].boxes:
                name = self.model.names[int(box.cls[0])]
                class_counts[name] = class_counts.get(name, 0) + 1

            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert(tk.END, "Objects Identified:\n------------------\n")
            for name, count in class_counts.items():
                self.stats_text.insert(tk.END, f"{name.capitalize()}: {count}\n")

            annotated = results[0].plot()
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((720, 580))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = SceneUnderstandingApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()