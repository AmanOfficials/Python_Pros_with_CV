import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk

class ExerciseCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 46: Exercise Repetition Counter (Curls)")
        self.root.geometry("900x700")

        self.cap = None
        self.is_running = False
        self.counter = 0
        self.stage = "down"

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Workout Tracker", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        self.lbl_reps = ttk.Label(toolbar, text="Reps: 0 | Stage: Down", font=("Arial", 12, "bold"))
        self.lbl_reps.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def calculate_angle(self, a, b, c):
        a = np.array(a) # Shoulder
        b = np.array(b) # Elbow
        c = np.array(c) # Wrist
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn.configure(text="Stop Tracker")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn.configure(text="Start Workout Tracker")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Right arm points: 12=Shoulder, 14=Elbow, 16=Wrist
                shoulder = [landmarks[12].x, landmarks[12].y]
                elbow = [landmarks[14].x, landmarks[14].y]
                wrist = [landmarks[16].x, landmarks[16].y]

                angle = self.calculate_angle(shoulder, elbow, wrist)

                # Rep counter state machine logic
                if angle > 160:
                    self.stage = "down"
                if angle < 40 and self.stage == "down":
                    self.stage = "up"
                    self.counter += 1

                self.lbl_reps.configure(text=f"Reps: {self.counter} | Stage: {self.stage.upper()}")

                # Draw joint angle on feed
                cv2.putText(frame, f"Angle: {int(angle)}", 
                            (int(elbow[0] * frame.shape[1]), int(elbow[1] * frame.shape[0])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_out)
            pil_img.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = ExerciseCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()