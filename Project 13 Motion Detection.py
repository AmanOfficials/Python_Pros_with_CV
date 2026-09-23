import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

class MotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 13: Motion Detection (Webcam)")
        self.root.geometry("850x650")

        self.cap = None
        self.is_running = False
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn_toggle = ttk.Button(toolbar, text="Start Camera", command=self.toggle_stream)
        self.btn_toggle.pack(side=tk.LEFT, padx=4)

        self.motion_status = ttk.Label(toolbar, text="Status: Inactive", foreground="gray", font=("Arial", 11, "bold"))
        self.motion_status.pack(side=tk.LEFT, padx=20)

        self.viewport = ttk.Label(self.root)
        self.viewport.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn_toggle.configure(text="Stop Camera")
                self.process_loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn_toggle.configure(text="Start Camera")
            self.motion_status.configure(text="Status: Inactive", foreground="gray")

    def process_loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            # Apply background subtractor
            mask = self.subtractor.apply(frame)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            for c in contours:
                if cv2.contourArea(c) > 1500: # Threshold area to discard small noise
                    motion_detected = True
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if motion_detected:
                self.motion_status.configure(text="Status: MOTION DETECTED", foreground="red")
            else:
                self.motion_status.configure(text="Status: Clear", foreground="green")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.viewport.configure(image=self.tk_photo)

        self.root.after(20, self.process_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = MotionDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()