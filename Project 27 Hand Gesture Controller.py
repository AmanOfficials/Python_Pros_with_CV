import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import pyautogui
from PIL import Image, ImageTk

class GestureControllerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 27: Hand Gesture Controller")
        self.root.geometry("850x650")

        self.cap = None
        self.is_running = False

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
        self.mp_draw = mp.solutions.drawing_utils

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Controller", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        self.gesture_lbl = ttk.Label(toolbar, text="Gesture: None", font=("Arial", 11, "bold"))
        self.gesture_lbl.pack(side=tk.LEFT, padx=20)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn.configure(text="Stop Controller")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn.configure(text="Start Controller")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            gesture = "Waiting..."
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    lms = hand_lms.landmark

                    # Detect Open Palm vs. Closed Fist
                    fingers_open = [lms[tip].y < lms[tip - 2].y for tip in [8, 12, 16, 20]]
                    if all(fingers_open):
                        gesture = "Next Slide / Vol Up"
                        # Trigger system key
                        pyautogui.press("right")
                    elif not any(fingers_open):
                        gesture = "Prev Slide / Vol Down"
                        # Trigger system key
                        pyautogui.press("left")

            self.gesture_lbl.configure(text=f"Gesture: {gesture}")

            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_out)
            pil_img.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        # Rate-limited loop to prevent keystroke spamming
        self.root.after(80, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureControllerApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()