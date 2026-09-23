import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk

class ContourApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 9: Contour Detection App")
        self.root.geometry("1000x700")

        self.img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=3)

        ttk.Label(toolbar, text="Draw Mode:").pack(side=tk.LEFT, padx=(10, 2))
        self.mode_cb = ttk.Combobox(toolbar, values=["Outlines", "Convex Hull", "Bounding Boxes"], state="readonly")
        self.mode_cb.current(0)
        self.mode_cb.bind("<<ComboboxSelected>>", self.render_contours)
        self.mode_cb.pack(side=tk.LEFT, padx=3)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            self.render_contours()

    def render_contours(self, _=None):
        if self.img is None:
            return

        output = self.img.copy()
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mode = self.mode_cb.get()

        for c in contours:
            if cv2.contourArea(c) < 100:
                continue

            if mode == "Outlines":
                cv2.drawContours(output, [c], -1, (0, 255, 0), 2)
            elif mode == "Convex Hull":
                hull = cv2.convexHull(c)
                cv2.drawContours(output, [hull], -1, (0, 0, 255), 2)
            elif mode == "Bounding Boxes":
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

        rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((900, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = ContourApp(root)
    root.mainloop()