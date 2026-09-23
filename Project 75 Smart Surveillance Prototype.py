import tkinter as tk
from tkinter import ttk
import cv2
import time
from PIL import Image, ImageTk

class SurveillanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 75: Smart Surveillance Prototype")
        self.root.geometry("850x650")

        self.cap = None
        self.is_running = False
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50)
        self.last_capture_time = 0

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Arm System", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        self.status = ttk.Label(toolbar, text="System: Disarmed", foreground="gray", font=("Arial", 11, "bold"))
        self.status.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn.configure(text="Disarm System")
                self.status.configure(text="System: ARMED & MONITORING", foreground="blue")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn.configure(text="Arm System")
            self.status.configure(text="System: Disarmed", foreground="gray")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            mask = self.subtractor.apply(frame)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            intruder_detected = False
            for c in cnts:
                if cv2.contourArea(c) > 3500:
                    intruder_detected = True
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if intruder_detected:
                self.status.configure(text="INTRUSION ALERT!", foreground="red")
                if time.time() - self.last_capture_time > 3.0:
                    cv2.imwrite(f"intruder_{int(time.time())}.jpg", frame)
                    self.last_capture_time = time.time()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = SurveillanceApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()