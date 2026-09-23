import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import pytesseract
from PIL import Image, ImageTk

# Point this to your Tesseract binary if not in PATH (e.g. C:\Program Files\Tesseract-OCR\tesseract.exe)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class PlateOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 24: License Plate Character Recognition")
        self.root.geometry("950x700")

        self.img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Plate/Car", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Run OCR", command=self.perform_ocr).pack(side=tk.LEFT, padx=5)

        self.result_lbl = ttk.Label(toolbar, text="Read Plate: [None]", font=("Arial", 12, "bold"))
        self.result_lbl.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            self.render(self.img)

    def perform_ocr(self):
        if self.img is None:
            return

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # Preprocessing: thresholding to clean background noise for OCR
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Configure Tesseract to look specifically for alphanumeric characters
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(thresh, config=custom_config).strip()

        self.result_lbl.configure(text=f"Read Plate: [{text if text else 'Not Recognized'}]")
        self.render(thresh)

    def render(self, image):
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((850, 550))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = PlateOCRApp(root)
    root.mainloop()