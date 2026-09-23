import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

class FaceCountingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 18: Face Counting System")
        self.root.geometry("850x650")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = None
        self.is_running = False

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Counter", command=self.toggle_camera)
        self.btn.pack(side=tk.LEFT, padx=5)

        self.status = ttk.Label(toolbar, text="Count: 0 | Capacity: Normal", font=("Arial", 11, "bold"))
        self.status.pack(side=tk.LEFT, padx=15)

        self.viewport = ttk.Label(self.root)
        self.viewport.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn.configure(text="Stop Counter")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn.configure(text="Start Counter")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
            count = len(faces)

            for i, (x, y, w, h) in enumerate(faces):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {i+1}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            color = "red" if count > 3 else "green"
            status_text = f"Count: {count} | " + ("OVERCROWDED!" if count > 3 else "Capacity OK")
            self.status.configure(text=status_text, foreground=color)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.viewport.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceCountingApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()