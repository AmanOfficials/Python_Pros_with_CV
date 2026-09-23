import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class StereoDepthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 91: Stereo Vision Depth Estimation")
        self.root.geometry("1000x700")

        self.img_left = None
        self.img_right = None

        # StereoSGBM block matcher
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 5,  # Must be divisible by 16
            blockSize=5,
            P1=8 * 3 * 5 ** 2,
            P2=32 * 3 * 5 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=32
        )

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Left Image", command=self.load_left).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Load Right Image", command=self.load_right).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Calculate Depth Map", command=self.compute_disparity).pack(side=tk.LEFT, padx=3)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_left(self):
        p = filedialog.askopenfilename()
        if p:
            self.img_left = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

    def load_right(self):
        p = filedialog.askopenfilename()
        if p:
            self.img_right = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

    def compute_disparity(self):
        if self.img_left is None or self.img_right is None:
            return

        # Resize right image if dimensions differ
        if self.img_left.shape != self.img_right.shape:
            self.img_right = cv2.resize(self.img_right, (self.img_left.shape[1], self.img_left.shape[0]))

        disparity = self.stereo.compute(self.img_left, self.img_right).astype(np.float32) / 16.0
        # Normalize disparity map for visualization
        norm_disp = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_depth = cv2.applyColorMap(norm_disp, cv2.COLORMAP_INFERNO)

        rgb = cv2.cvtColor(color_depth, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((900, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = StereoDepthApp(root)
    root.mainloop()