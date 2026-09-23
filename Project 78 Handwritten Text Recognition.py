import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import pytesseract
from PIL import Image, ImageTk

class HandwrittenTextRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 78: Handwritten Text Recognition")
        self.root.geometry("1000x650")

        self.img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Handwritten Sample", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Transcribe", command=self.transcribe).pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.txt_out = tk.Text(self.root, width=40, font=("Helvetica", 11))
        self.txt_out.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((600, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

    def transcribe(self):
        if self.img is None:
            return

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # Apply bilateral filter to smooth paper fiber while retaining ink strokes
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Page segmentation mode 6 assumes a uniform block of text
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        self.txt_out.delete("1.0", tk.END)
        self.txt_out.insert(tk.END, text if text.strip() else "[No text detected]")

if __name__ == "__main__":
    root = tk.Tk()
    app = HandwrittenTextRecognitionApp(root)
    root.mainloop()