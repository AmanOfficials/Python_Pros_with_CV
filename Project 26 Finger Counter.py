import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
from PIL import Image, ImageTk

class FingerCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 26: Finger Counter")
        self.root.geometry("850x650")

        self.cap = None
        self.is_running = False

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Feed", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        self.count_lbl = ttk.Label(toolbar, text="Fingers Raised: 0", font=("Arial", 12, "bold"))
        self.count_lbl.pack(side=tk.LEFT, padx=20)

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

            finger_count = 0
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    lm_list = [[id, int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])] for id, lm in enumerate(hand_lms.landmark)]

                    if len(lm_list) != 0:
                        fingers = []
                        # Thumb (horizontal orientation check)
                        if lm_list[self.tip_ids[0]][1] > lm_list[self.tip_ids[0] - 1][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                        # 4 Fingers: Check if tip Y is higher (smaller value) than PIP joint Y
                        for id in range(1, 5):
                            if lm_list[self.tip_ids[id]][2] < lm_list[self.tip_ids[id] - 2][2]:
                                fingers.append(1)
                            else:
                                fingers.append(0)

                        finger_count = fingers.count(1)

            self.count_lbl.configure(text=f"Fingers Raised: {finger_count}")
            cv2.putText(frame, str(finger_count), (45, 100), cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 0), 5)

            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_out)
            pil_img.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = FingerCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()