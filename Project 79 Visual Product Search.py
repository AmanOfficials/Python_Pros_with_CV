import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ProductSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 79: Visual Product Search")
        self.root.geometry("1000x650")

        self.query_img = None
        self.candidate_img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Query Product", command=self.load_query).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Load Catalog Candidate", command=self.load_candidate).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Calculate Match Score", command=self.match_products).pack(side=tk.LEFT, padx=4)

        self.score_lbl = ttk.Label(toolbar, text="Match Score: 0%", font=("Arial", 12, "bold"))
        self.score_lbl.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_query(self):
        p = filedialog.askopenfilename()
        if p:
            self.query_img = cv2.imread(p)

    def load_candidate(self):
        p = filedialog.askopenfilename()
        if p:
            self.candidate_img = cv2.imread(p)

    def extract_features(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [18, 25], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def match_products(self):
        if self.query_img is None or self.candidate_img is None:
            return

        f1 = self.extract_features(self.query_img)
        f2 = self.extract_features(self.candidate_img)

        # Correlation metric: 1.0 indicates a perfect match
        score = cv2.compareHist(f1, f2, cv2.HISTCMP_CORREL)
        match_pct = max(0.0, score * 100)
        self.score_lbl.configure(text=f"Match Score: {match_pct:.1f}%")

        # Side-by-side thumbnail visualization
        h, w = 300, 300
        q_thumb = cv2.resize(self.query_img, (w, h))
        c_thumb = cv2.resize(self.candidate_img, (w, h))
        combined = np.hstack([q_thumb, c_thumb])

        rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = ProductSearchApp(root)
    root.mainloop()