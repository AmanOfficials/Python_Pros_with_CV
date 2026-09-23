import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class VirtualDrawingBoardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 15: Virtual Drawing Board")
        self.root.geometry("900x700")

        self.cap = None
        self.is_running = False
        self.canvas_layer = None
        self.prev_point = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.toggle_btn = ttk.Button(toolbar, text="Start Camera", command=self.toggle_camera)
        self.toggle_btn.pack(side=tk.LEFT, padx=3)

        ttk.Button(toolbar, text="Clear Canvas", command=self.clear_canvas).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Save Art", command=self.save_artwork).pack(side=tk.LEFT, padx=3)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.toggle_btn.configure(text="Stop Camera")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.toggle_btn.configure(text="Start Camera")

    def clear_canvas(self):
        self.canvas_layer = None

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            if self.canvas_layer is None:
                self.canvas_layer = np.zeros((h, w, 3), dtype=np.uint8)

            # Detect blue pointer/marker (HSV range)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([100, 150, 100])
            upper_blue = np.array([140, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 300:
                    ((x, y), _) = cv2.minEnclosingCircle(c)
                    curr_pt = (int(x), int(y))

                    if self.prev_point is not None:
                        cv2.line(self.canvas_layer, self.prev_point, curr_pt, (0, 255, 255), 5)
                    self.prev_point = curr_pt
            else:
                self.prev_point = None

            # Merge drawing layer with live camera frame
            combined = cv2.addWeighted(frame, 0.8, self.canvas_layer, 1.0, 0)

            rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

    def save_artwork(self):
        if self.canvas_layer is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                cv2.imwrite(path, self.canvas_layer)

if __name__ == "__main__":
    root = tk.Tk()
    app = VirtualDrawingBoardApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()