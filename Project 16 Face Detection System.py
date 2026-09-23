import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 16: Face Detection System")
        self.root.geometry("900x650")

        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = None
        self.is_running = False

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=3)
        self.cam_btn = ttk.Button(toolbar, text="Toggle Webcam", command=self.toggle_camera)
        self.cam_btn.pack(side=tk.LEFT, padx=3)

        self.count_lbl = ttk.Label(toolbar, text="Faces: 0", font=("Arial", 11, "bold"))
        self.count_lbl.pack(side=tk.LEFT, padx=15)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def detect_and_draw(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        self.count_lbl.configure(text=f"Faces: {len(faces)}")

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame

    def load_image(self):
        self.stop_camera()
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path)
            processed = self.detect_and_draw(img)
            self.render(processed)

    def toggle_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.cam_btn.configure(text="Stop Camera")
                self.stream()
        else:
            self.stop_camera()

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.cam_btn.configure(text="Toggle Webcam")

    def stream(self):
        if not self.is_running:
            return
        ret, frame = self.cap.read()
        if ret:
            processed = self.detect_and_draw(frame)
            self.render(processed)
        self.root.after(20, self.stream)

    def render(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((850, 550))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()