import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

class HeadPoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 31: Head Pose Estimation")
        self.root.geometry("900x700")

        self.cap = None
        self.is_running = False

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Feed", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        self.status = ttk.Label(toolbar, text="Orientation: None", font=("Arial", 11, "bold"))
        self.status.pack(side=tk.LEFT, padx=20)

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

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_2d = []
                    face_3d = []

                    # Key anchor indices for PnP alignment
                    key_points = [33, 263, 1, 61, 291, 199]
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in key_points:
                            x, y = int(lm.x * w), int(lm.y * h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = 1 * w
                    cam_matrix = np.array([[focal_length, 0, h / 2],
                                           [0, focal_length, w / 2],
                                           [0, 0, 1]])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360

                    if y < -10:
                        text = "Looking Left"
                    elif y > 10:
                        text = "Looking Right"
                    elif x < -8:
                        text = "Looking Down"
                    elif x > 10:
                        text = "Looking Up"
                    else:
                        text = "Forward"

                    self.status.configure(text=f"Orientation: {text} (Pitch: {int(x)}, Yaw: {int(y)})")

                    # Project nose coordinate line
                    nose_2d = (int(face_landmarks.landmark[1].x * w), int(face_landmarks.landmark[1].y * h))
                    p1 = nose_2d
                    p2 = (int(nose_2d[0] + y * 2), int(nose_2d[1] - x * 2))
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)

            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_out)
            pil_img.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = HeadPoseApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()