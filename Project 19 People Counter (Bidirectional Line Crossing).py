import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

class PeopleCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 19: People Counter (Line Crossing)")
        self.root.geometry("850x650")

        self.cap = None
        self.is_running = False
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40)
        
        self.in_count = 0
        self.out_count = 0
        self.tracked_objects = {}

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Video/Camera", command=self.toggle_camera)
        self.btn.pack(side=tk.LEFT, padx=5)

        self.lbl_stats = ttk.Label(toolbar, text="IN: 0 | OUT: 0", font=("Arial", 11, "bold"))
        self.lbl_stats.pack(side=tk.LEFT, padx=15)

        self.viewport = ttk.Label(self.root)
        self.viewport.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn.configure(text="Stop Stream")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn.configure(text="Start Video/Camera")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            h, w, _ = frame.shape
            line_y = int(h * 0.5)

            # Draw virtual reference counting line
            cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)

            mask = self.subtractor.apply(frame)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                if cv2.contourArea(c) < 2500:
                    continue
                (x, y, cw, ch) = cv2.boundingRect(c)
                cy = int(y + ch / 2)
                cx = int(x + cw / 2)

                cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

                # Track vertical transitions over virtual line
                if abs(cy - line_y) < 15:
                    if y < line_y:
                        self.in_count += 1
                    else:
                        self.out_count += 1

            self.lbl_stats.configure(text=f"IN: {self.in_count} | OUT: {self.out_count}")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.viewport.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = PeopleCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()