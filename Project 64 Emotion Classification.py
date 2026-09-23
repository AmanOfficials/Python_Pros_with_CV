import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

class EmotionClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 64: Real-Time Emotion Classification")
        self.root.geometry("850x650")

        self.cap = None
        self.is_running = False

        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.6)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Feed", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        self.lbl_emotion = ttk.Label(toolbar, text="Emotion: Neutral", font=("Arial", 12, "bold"))
        self.lbl_emotion.pack(side=tk.LEFT, padx=20)

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
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            emotion = "Neutral"
            if results.multi_face_landmarks:
                for fl in results.multi_face_landmarks:
                    lms = fl.landmark
                    # Landmark 13: Upper Lip, 14: Lower Lip, 61: Left Mouth Corner, 291: Right Mouth Corner
                    lip_dist = np.hypot(lms[13].x - lms[14].x, lms[13].y - lms[14].y) * h
                    mouth_width = np.hypot(lms[61].x - lms[291].x, lms[61].y - lms[291].y) * w

                    if lip_dist > 25:
                        emotion = "Surprised / Speaking"
                    elif mouth_width > 85:
                        emotion = "Happy / Smiling"
                    else:
                        emotion = "Neutral / Focused"

            self.lbl_emotion.configure(text=f"Emotion: {emotion}")
            cv2.putText(frame, emotion, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_out)
            pil_img.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionClassifierApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()