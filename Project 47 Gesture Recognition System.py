import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
from PIL import Image, ImageTk

class GestureRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 47: Static Gesture Recognition")
        self.root.geometry("850x650")

        self.cap = None
        self.is_running = False

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Feed", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        self.lbl_gesture = ttk.Label(toolbar, text="Recognized: None", font=("Arial", 12, "bold"))
        self.lbl_gesture.pack(side=tk.LEFT, padx=20)

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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            gesture = "Unknown"
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    lms = hand_lms.landmark
                    # Finger states (True if tip Y is above PIP Y)
                    index_open = lms[8].y < lms[6].y
                    middle_open = lms[12].y < lms[10].y
                    ring_open = lms[16].y < lms[14].y
                    pinky_open = lms[20].y < lms[18].y
                    thumb_up = lms[4].y < lms[3].y

                    if index_open and middle_open and not ring_open and not pinky_open:
                        gesture = "Peace / Victory (V)"
                    elif all([index_open, middle_open, ring_open, pinky_open]):
                        gesture = "Open Palm (Stop)"
                    elif not any([index_open, middle_open, ring_open, pinky_open]):
                        gesture = "Fist"
                    elif thumb_up and not any([index_open, middle_open, ring_open, pinky_open]):
                        gesture = "Thumbs Up"

            self.lbl_gesture.configure(text=f"Recognized: {gesture}")

            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_out)
            pil_img.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(30, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()