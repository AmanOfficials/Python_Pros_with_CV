import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class HistogramAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 10: Image Histogram Analyzer")
        self.root.geometry("1100x650")

        self.img = None

        top_frame = ttk.Frame(self.root, padding=6)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(top_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)

        # Left split: Image preview; Right split: Matplotlib plot
        self.img_label = ttk.Label(self.root)
        self.img_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.plot_canvas.get_tk_widget().pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            self.display_image_and_histogram()

    def display_image_and_histogram(self):
        if self.img is None:
            return

        # Render preview image
        rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((450, 450))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.img_label.configure(image=self.tk_photo)

        # Compute and render histogram plots
        self.ax.clear()
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([self.img], [i], None, [256], [0, 256])
            self.ax.plot(hist, color=col)
            self.ax.set_xlim([0, 256])

        self.ax.set_title("Color Distribution")
        self.ax.set_xlabel("Pixel Intensity")
        self.ax.set_ylabel("Frequency")
        self.plot_canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = HistogramAnalyzerApp(root)
    root.mainloop()