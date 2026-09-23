import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk

class ImageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 3: Image Filtering App")
        self.root.geometry("1000x700")

        self.original_img = None
        self.processed_img = None

        # Control panel
        panel = ttk.Frame(self.root, padding=10)
        panel.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(panel, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)

        ttk.Label(panel, text="Filter:").pack(side=tk.LEFT, padx=(10, 2))
        self.filter_type = ttk.Combobox(panel, values=["Box Filter", "Gaussian Blur", "Median Blur", "Bilateral Filter"], state="readonly")
        self.filter_type.current(1)
        self.filter_type.bind("<<ComboboxSelected>>", self.apply_filter)
        self.filter_type.pack(side=tk.LEFT)

        ttk.Label(panel, text="Kernel Size:").pack(side=tk.LEFT, padx=(15, 2))
        self.kernel_slider = ttk.Scale(panel, from_=1, to=31, value=5, command=self.apply_filter)
        self.kernel_slider.pack(side=tk.LEFT, padx=5)

        self.viewport = ttk.Label(self.root)
        self.viewport.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.original_img = cv2.imread(path)
            self.apply_filter()

    def apply_filter(self, _=None):
        if self.original_img is None:
            return

        # Kernel sizes must be odd numbers > 0
        k = int(self.kernel_slider.get())
        if k % 2 == 0:
            k += 1

        selected = self.filter_type.get()
        if selected == "Box Filter":
            self.processed_img = cv2.blur(self.original_img, (k, k))
        elif selected == "Gaussian Blur":
            self.processed_img = cv2.GaussianBlur(self.original_img, (k, k), 0)
        elif selected == "Median Blur":
            self.processed_img = cv2.medianBlur(self.original_img, k)
        elif selected == "Bilateral Filter":
            # Bilateral keeps edges sharp while smoothing flat areas
            self.processed_img = cv2.bilateralFilter(self.original_img, d=k, sigmaColor=75, sigmaSpace=75)

        rgb = cv2.cvtColor(self.processed_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((900, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.viewport.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageFilterApp(root)
    root.mainloop()