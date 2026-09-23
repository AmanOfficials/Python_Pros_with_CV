import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk

class ImageSimilarityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 55: Image Similarity Analyzer")
        self.root.geometry("1100x650")

        self.img1 = None
        self.img2 = None
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image A", command=self.load_img1).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Load Image B", command=self.load_img2).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Compare Similarity", command=self.compare_images).pack(side=tk.LEFT, padx=3)

        self.score_lbl = ttk.Label(toolbar, text="Similarity: 0.0%", font=("Arial", 11, "bold"))
        self.score_lbl.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_img1(self):
        p = filedialog.askopenfilename()
        if p:
            self.img1 = cv2.imread(p)

    def load_img2(self):
        p = filedialog.askopenfilename()
        if p:
            self.img2 = cv2.imread(p)

    def compare_images(self):
        if self.img1 is None or self.img2 is None:
            return

        kp1, des1 = self.orb.detectAndCompute(self.img1, None)
        kp2, des2 = self.orb.detectAndCompute(self.img2, None)

        if des1 is not None and des2 is not None:
            matches = self.matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Similarity calculation based on top matching descriptors
            good_matches = [m for m in matches if m.distance < 50]
            similarity = (len(good_matches) / max(len(kp1), len(kp2))) * 100
            self.score_lbl.configure(text=f"Similarity: {min(similarity, 100.0):.1f}%")

            match_img = cv2.drawMatches(self.img1, kp1, self.img2, kp2, matches[:30], None, flags=2)
            rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((950, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSimilarityApp(root)
    root.mainloop()