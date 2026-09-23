import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from PIL import Image, ImageTk

# Feedforward MLP to classify coordinates
class PoseMLP(nn.Module):
    def __init__(self):
        super(PoseMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(66, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # Standing, Sitting, Lying
        )
    def forward(self, x):
        return self.classifier(x)

class PoseClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 70: Pose Classification (MLP)")
        self.root.geometry("850x650")

        self.cap = None
        self.is_running = False

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.6)
        self.mp_draw = mp.solutions.drawing_utils

        self.mlp = PoseMLP()
        self.mlp.eval()
        self.classes = ["Standing", "Sitting", "Lying / Reclined"]

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Video Feed", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        self.lbl_class = ttk.Label(toolbar, text="Posture: None", font=("Arial", 12, "bold"))
        self.lbl_class.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn.configure(text="Stop Video Feed")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn.configure(text="Start Video Feed")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            posture = "Unknown"
            if results.pose_landmarks:
                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Extract (x, y) landmark feature vector
                feat = []
                for lm in results.pose_landmarks.landmark:
                    feat.extend([lm.x, lm.y])

                tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = self.mlp(tensor)
                    pred_idx = torch.argmax(logits, dim=1).item()
                    posture = self.classes[pred_idx]

            self.lbl_class.configure(text=f"Posture: {posture}")
            cv2.putText(frame, posture, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_out)
            pil_img.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = PoseClassifierApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()