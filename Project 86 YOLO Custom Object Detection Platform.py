import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk

class YOLOAnnotationPlatformApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 86: YOLO Custom Annotation Platform")
        self.root.geometry("1000x750")

        self.img = None
        self.raw_path = None
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.boxes = []

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Training Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Export YOLO .txt", command=self.export_annotations).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Clear Boxes", command=self.clear_boxes).pack(side=tk.LEFT, padx=4)

        self.canvas = tk.Canvas(self.root, bg="#1e1e1e")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def load_image(self):
        self.raw_path = filedialog.askopenfilename()
        if self.raw_path:
            self.img = cv2.imread(self.raw_path)
            self.h, self.w, _ = self.img.shape
            self.render()

    def render(self):
        rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.pil_img = Image.fromarray(rgb)
        self.tk_img = ImageTk.PhotoImage(self.pil_img)
        self.canvas.config(width=self.w, height=self.h)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="green", width=2)

    def on_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        x1, y1, x2, y2 = min(self.start_x, event.x), min(self.start_y, event.y), max(self.start_x, event.x), max(self.start_y, event.y)
        if x2 - x1 > 5 and y2 - y1 > 5:
            # Normalize to YOLO format: class_id, x_center, y_center, w, h (0.0 to 1.0)
            xc = ((x1 + x2) / 2.0) / self.w
            yc = ((y1 + y2) / 2.0) / self.h
            bw = (x2 - x1) / float(self.w)
            bh = (y2 - y1) / float(self.h)
            self.boxes.append((0, xc, yc, bw, bh))

    def export_annotations(self):
        if not self.boxes or not self.raw_path:
            return
        out_txt = self.raw_path.rsplit(".", 1)[0] + ".txt"
        with open(out_txt, "w") as f:
            for b in self.boxes:
                f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
        tk.messagebox.showinfo("Export Complete", f"Saved YOLO labels to {out_txt}")

    def clear_boxes(self):
        self.boxes = []
        if self.img is not None:
            self.render()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOAnnotationPlatformApp(root)
    root.mainloop()