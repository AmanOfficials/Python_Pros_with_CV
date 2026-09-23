import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class CrowdAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 53: Crowd Density Analysis")
        self.root.geometry("950x700")

        self.cap = None
        self.is_running = False
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=30)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Video", command=self.load_video).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Webcam", command=lambda: self.start_stream(0)).pack(side=tk.LEFT, padx=4)

        self.lbl_density = ttk.Label(toolbar, text="Crowd Level: Low", font=("Arial", 11, "bold"))
        self.lbl_density.pack(side=tk.LEFT, padx=15)

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
            fgmask = self.fgbg.apply(frame)
            # Smooth mask to form thermal density maps
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            dilated = cv2.dilate(fgmask, kernel, iterations=2)
            blurred = cv2.GaussianBlur(dilated, (25, 25), 0)

            # Apply pseudo-color heatmap
            heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame, 0.65, heatmap, 0.35, 0)

            density_val = np.sum(fgmask > 0) / (frame.shape[0] * frame.shape[1])
            crowd_status = "High (Congested)" if density_val > 0.15 else "Moderate" if density_val > 0.05 else "Low"
            self.lbl_density.configure(text=f"Crowd Level: {crowd_status}")

            rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

            self.root.after(20, self.loop)
        else:
            self.is_running = False

if __name__ == "__main__":
    root = tk.Tk()
    app = CrowdAnalysisApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()