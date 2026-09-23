import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk

class GrayscaleConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 2: Grayscale Image Converter")
        self.root.geometry("950x650")

        self.cv_img = None
        self.mode = "Standard Grayscale"

        # Toolbar controls
        top_bar = ttk.Frame(self.root, padding=8)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top_bar, text="Open Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_bar, text="Save Image", command=self.save_image).pack(side=tk.LEFT, padx=5)

        ttk.Label(top_bar, text="Conversion Mode:").pack(side=tk.LEFT, padx=(15, 5))
        self.channel_box = ttk.Combobox(top_bar, values=[
            "Standard Grayscale", 
            "Blue Channel Only", 
            "Green Channel Only", 
            "Red Channel Only"
        ], state="readonly")
        self.channel_box.current(0)
        self.channel_box.bind("<<ComboboxSelected>>", self.convert_and_display)
        self.channel_box.pack(side=tk.LEFT)

        # Image display container
        self.display_label = ttk.Label(self.root)
        self.display_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.cv_img = cv2.imread(path)
            self.convert_and_display()

    def convert_and_display(self, _=None):
        if self.cv_img is None:
            return

        choice = self.channel_box.get()
        # Split BGR channels
        b, g, r = cv2.split(self.cv_img)

        if choice == "Standard Grayscale":
            # ITU-R 601-2 luma formula: Y = 0.299 R + 0.587 G + 0.114 B
            processed = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)
        elif choice == "Blue Channel Only":
            processed = b
        elif choice == "Green Channel Only":
            processed = g
        elif choice == "Red Channel Only":
            processed = r

        self.processed_output = processed
        pil_img = Image.fromarray(processed)
        pil_img.thumbnail((850, 550))

        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display_label.configure(image=self.tk_photo)

    def save_image(self):
        if hasattr(self, 'processed_output'):
            save_path = filedialog.asksaveasfilename(defaultextension=".png")
            if save_path:
                cv2.imwrite(save_path, self.processed_output)

if __name__ == "__main__":
    root = tk.Tk()
    app = GrayscaleConverterApp(root)
    root.mainloop()