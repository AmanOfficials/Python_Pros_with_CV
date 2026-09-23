import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk

class LicensePlateDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 23: License Plate Detection")
        self.root.geometry("950x650")

        self.img = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Vehicle Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        self.status = ttk.Label(toolbar, text="Upload an image to locate plate candidate.", font=("Arial", 10))
        self.status.pack(side=tk.LEFT, padx=15)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path)
            self.detect_plate()

    def detect_plate(self):
        if self.img is None:
            return

        output = self.img.copy()
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(blur, 30, 200)

        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        plate_found = False
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)

            # Plates are generally 4-point quadrilaterals with broad aspect ratios
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 2.0 <= aspect_ratio <= 6.0:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(output, "Plate Region", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    plate_found = True
                    break

        self.status.configure(text="Plate Located!" if plate_found else "No standard plate identified.")

        rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((900, 550))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateDetectorApp(root)
    root.mainloop()