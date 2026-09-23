import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

class VirtualKeyboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 29: Virtual Keyboard")
        self.root.geometry("900x700")

        self.cap = None
        self.is_running = False
        self.typed_text = ""

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

        # Virtual keys grid
        self.keys = [
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
            ["Z", "X", "C", "V", "B", "N", "M", ",", ".", " "]
        ]

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Feed", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        ttk.Button(toolbar, text="Clear", command=self.clear_text).pack(side=tk.LEFT, padx=4)

        self.output_lbl = ttk.Label(toolbar, text="Output: ", font=("Arial", 12, "bold"))
        self.output_lbl.pack(side=tk.LEFT, padx=15)

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

    def clear_text(self):
        self.typed_text = ""
        self.output_lbl.configure(text="Output: ")

    def draw_keyboard(self, frame):
        for row_idx, row in enumerate(self.keys):
            for col_idx, key in enumerate(row):
                x = 50 + col_idx * 55
                y = 50 + row_idx * 55
                cv2.rectangle(frame, (x, y), (x + 50, y + 50), (200, 200, 200), -1)
                cv2.putText(frame, key, (x + 15, y + 35), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        return frame

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = self.draw_keyboard(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    ix = int(hand_lms.landmark[8].x * frame.shape[1])
                    iy = int(hand_lms.landmark[8].y * frame.shape[0])
                    tx = int(hand_lms.landmark[4].x * frame.shape[1])
                    ty = int(hand_lms.landmark[4].y * frame.shape[0])

                    cv2.circle(frame, (ix, iy), 6, (0, 0, 255), -1)

                    # Pinch gesture: Trigger keypress
                    if np.hypot(ix - tx, iy - ty) < 30:
                        for row_idx, row in enumerate(self.keys):
                            for col_idx, key in enumerate(row):
                                kx = 50 + col_idx * 55
                                ky = 50 + row_idx * 55
                                if kx < ix < kx + 50 and ky < iy < ky + 50:
                                    cv2.rectangle(frame, (kx, ky), (kx + 50, ky + 50), (0, 255, 0), -1)
                                    self.typed_text += key
                                    self.output_lbl.configure(text=f"Output: {self.typed_text}")

            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_out)
            pil_img.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(35, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = VirtualKeyboardApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()