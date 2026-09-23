import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import pytesseract
from PIL import Image, ImageTk

class AIOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 77: AI Optical Character Recognition (OCR)")
        self.root.geometry("1100x700")

        self.img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Execute OCR", command=self.run_ocr).pack(side=tk.LEFT, padx=4)

        # Split: Image preview on left, extracted text editor on right
        content = ttk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.display = ttk.Label(content)
        self.display.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.text_editor = tk.Text(content, width=45, wrap=tk.WORD)
        self.text_editor.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((650, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

    def run_ocr(self):
        if self.img is None:
            return

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # Clean background noise using Otsu binarization
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        extracted_text = pytesseract.image_to_string(thresh)

        self.text_editor.delete("1.0", tk.END)
        self.text_editor.insert(tk.END, extracted_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = AIOCRApp(root)
    root.mainloop()