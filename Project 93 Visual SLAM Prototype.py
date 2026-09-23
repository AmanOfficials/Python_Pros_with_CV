import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class VisualSLAMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 93: Visual SLAM / Odometry Prototype")
        self.root.geometry("1050x650")

        self.cap = None
        self.is_running = False
        self.orb = cv2.ORB_create(3000)
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        self.trajectory = np.zeros((600, 400, 3), dtype=np.uint8)
        self.pos = np.array([200, 300])

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Odometry", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        # Split: Left viewport for feature points, Right viewport for mapped trajectory
        split = ttk.Frame(self.root)
        split.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.cam_viewport = ttk.Label(split)
        self.cam_viewport.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.map_viewport = ttk.Label(split)
        self.map_viewport.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=(10, 0))

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn.configure(text="Stop SLAM")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn.configure(text="Start Odometry")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(gray, None)

            if self.prev_des is not None and des is not None and len(des) > 0 and len(self.prev_des) > 0:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(self.prev_des, des)

                if len(matches) > 15:
                    # Estimate translational movement vector
                    pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
                    pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])
                    motion = np.mean(pts_curr - pts_prev, axis=0)

                    # Update mapped trajectory
                    new_pos = self.pos + np.array([int(motion[0] * 0.5), int(motion[1] * 0.5)])
                    new_pos = np.clip(new_pos, [10, 10], [390, 590])
                    cv2.line(self.trajectory, tuple(self.pos), tuple(new_pos), (0, 255, 0), 2)
                    self.pos = new_pos

            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des

            frame_kps = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 255), flags=0)

            # Update Left feed
            rgb_f = cv2.cvtColor(frame_kps, cv2.COLOR_BGR2RGB)
            pil_f = Image.fromarray(rgb_f)
            pil_f.thumbnail((500, 450))
            self.tk_f = ImageTk.PhotoImage(pil_f)
            self.cam_viewport.configure(image=self.tk_f)

            # Update Right map
            rgb_m = cv2.cvtColor(self.trajectory, cv2.COLOR_BGR2RGB)
            pil_m = Image.fromarray(rgb_m)
            self.tk_m = ImageTk.PhotoImage(pil_m)
            self.map_viewport.configure(image=self.tk_m)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = VisualSLAMApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()