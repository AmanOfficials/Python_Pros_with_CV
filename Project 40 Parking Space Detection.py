import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ParkingSpaceDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 40: Parking Space Detection")
        self.root.geometry("900x700")

        # Mock predefined parking spot bounding boxes: [(x, y, w, h), ...]
        self.parking_slots = [
            (50, 200, 100, 180),
            (170, 200, 100, 180),
            (290, 200, 100, 180),
            (410, 200, 100, 180),
            (530, 200, 100, 180)
        ]

        self.cap = None
        self.is_running = False

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Feed", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        self.status = ttk.Label(toolbar, text="Free Slots: 0 / 5", font=("Arial", 11, "bold"))
        self.status.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn.configure(text="Stop Feed")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn.configure(text="Start Feed")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 1)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)

            free_slots = 0
            for (x, y, w, h) in self.parking_slots:
                # Ensure coordinates are within frame dimensions
                if x + w < frame.shape[1] and y + h < frame.shape[0]:
                    slot_crop = thresh[y:y+h, x:x+w]
                    count = cv2.countNonZero(slot_crop)

                    # Threshold determines if slot contains a vehicle
                    if count < 900:
                        color = (0, 255, 0)
                        free_slots += 1
                        label = "Empty"
                    else:
                        color = (0, 0, 255)
                        label = "Occupied"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            self.status.configure(text=f"Free Slots: {free_slots} / {len(self.parking_slots)}")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(30, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = ParkingSpaceDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()