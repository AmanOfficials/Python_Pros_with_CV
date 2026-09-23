import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class VisualNavigationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 84: Visual Navigation & Obstacle Avoidance")
        self.root.geometry("900x700")

        self.cap = None
        self.is_running = False

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Navigation", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        self.nav_lbl = ttk.Label(toolbar, text="Guidance: Ready", font=("Arial", 12, "bold"))
        self.nav_lbl.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn.configure(text="Stop Navigation")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn.configure(text="Start Navigation")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 40, 120)

            # Divide perspective field into Left, Center, and Right zones
            third = w // 3
            left_zone = edges[:, :third]
            center_zone = edges[:, third:2*third]
            right_zone = edges[:, 2*third:]

            counts = [cv2.countNonZero(left_zone), cv2.countNonZero(center_zone), cv2.countNonZero(right_zone)]
            
            # Draw boundary zone lines
            cv2.line(frame, (third, 0), (third, h), (255, 255, 0), 2)
            cv2.line(frame, (2 * third, 0), (2 * third, h), (255, 255, 0), 2)

            # Determine navigational recommendation
            if counts[1] > 2500:
                # Obstacle in center path: guide toward least dense side
                if counts[0] < counts[2]:
                    nav_text = "OBSTACLE AHEAD: STEER LEFT"
                    color = (0, 0, 255)
                else:
                    nav_text = "OBSTACLE AHEAD: STEER RIGHT"
                    color = (0, 0, 255)
            else:
                nav_text = "PATH CLEAR: MOVE STRAIGHT"
                color = (0, 255, 0)

            self.nav_lbl.configure(text=f"Guidance: {nav_text}")
            cv2.putText(frame, nav_text, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = VisualNavigationApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()