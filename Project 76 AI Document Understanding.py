import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk

class DocumentUnderstandingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 76: AI Document Understanding")
        self.root.geometry("950x700")

        self.img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Document", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Extract Layout Structure", command=self.parse_layout).pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            self.render(self.img)

    def parse_layout(self):
        if self.img is None:
            return

        frame = self.img.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)

        # Merge horizontal lines of text into solid blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 4))
        dilated = cv2.dilate(thresh, kernel, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 40 and h > 15:
                # Classify block type based on aspect ratio
                if h > 120 and w > 120:
                    label = "Figure / Table"
                    color = (255, 0, 0)
                elif h < 40 and w > 200:
                    label = "Heading / Title"
                    color = (0, 0, 255)
                else:
                    label = "Paragraph"
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        self.render(frame)

    def render(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((850, 600))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentUnderstandingApp(root)
    root.mainloop()