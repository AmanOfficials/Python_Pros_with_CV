import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from PIL import Image, ImageTk

# Prevent PyAutoGUI failsafe crash on screen corners
pyautogui.FAILSAFE = False

class VirtualMouseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 28: Virtual Mouse")
        self.root.geometry("850x650")

        self.cap = None
        self.is_running = False

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.screen_w, self.screen_h = pyautogui.size()

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Enable Mouse", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn.configure(text="Disable Mouse")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn.configure(text="Enable Mouse")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    # Index fingertip: Landmark 8
                    ix = int(hand_lms.landmark[8].x * w)
                    iy = int(hand_lms.landmark[8].y * h)

                    # Thumb tip: Landmark 4
                    tx = int(hand_lms.landmark[4].x * w)
                    ty = int(hand_lms.landmark[4].y * h)

                    # Map coordinates to desktop dimensions
                    target_x = np.interp(ix, [50, w - 50], [0, self.screen_w])
                    target_y = np.interp(iy, [50, h - 50], [0, self.screen_h])

                    pyautogui.moveTo(target_x, target_y)

                    # Measure pinch distance between index and thumb for click
                    dist = np.hypot(ix - tx, iy - ty)
                    cv2.circle(frame, (ix, iy), 8, (255, 0, 0), -1)

                    if dist < 25:
                        cv2.circle(frame, (ix, iy), 12, (0, 255, 0), -1)
                        pyautogui.click()

            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_out)
            pil_img.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(15, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = VirtualMouseApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()