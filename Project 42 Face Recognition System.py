import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 42: LBPH Face Recognition")
        self.root.geometry("850x650")

        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.names = {}
        self.trained = False

        self.cap = None
        self.is_running = False

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn_cam = ttk.Button(toolbar, text="Start Video", command=self.toggle_cam)
        self.btn_cam.pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Enroll New Face", command=self.enroll_face).pack(side=tk.LEFT, padx=3)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_cam(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn_cam.configure(text="Stop Video")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn_cam.configure(text="Start Video")

    def enroll_face(self):
        name = simpledialog.askstring("Enroll Face", "Enter User Name:")
        if not name:
            return

        messagebox.showinfo("Instructions", "Look at the camera while 30 face samples are captured.")
        temp_cap = cv2.VideoCapture(0)
        samples = []
        labels = []
        user_id = len(self.names) + 1
        self.names[user_id] = name

        count = 0
        while count < 30:
            ret, frame = temp_cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                face_crop = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                samples.append(face_crop)
                labels.append(user_id)
                count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("Enrolling... Hold steady", frame)
            cv2.waitKey(50)

        temp_cap.release()
        cv2.destroyAllWindows()

        if len(samples) > 0:
            if not self.trained:
                self.recognizer.train(samples, np.array(labels))
                self.trained = True
            else:
                self.recognizer.update(samples, np.array(labels))
            messagebox.showinfo("Success", f"Enrolled {name} successfully!")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                label_text = "Unknown"
                color = (0, 0, 255)

                if self.trained:
                    face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                    label_id, confidence = self.recognizer.predict(face_roi)
                    # Lower confidence in LBPH means closer mathematical match
                    if confidence < 75:
                        label_text = f"{self.names.get(label_id, 'User')} ({int(confidence)})"
                        color = (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label_text, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()