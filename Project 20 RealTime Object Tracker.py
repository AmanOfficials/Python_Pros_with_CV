import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

class RealTimeTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 20: Real-Time Object Tracker")
        self.root.geometry("850x650")

        self.cap = None
        self.is_running = False
        self.tracker = None
        self.is_tracking = False

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.cam_btn = ttk.Button(toolbar, text="Start Webcam", command=self.toggle_camera)
        self.cam_btn.pack(side=tk.LEFT, padx=3)

        ttk.Button(toolbar, text="Select Target Box", command=self.select_roi).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Reset Tracker", command=self.reset_tracker).pack(side=tk.LEFT, padx=3)

        self.status = ttk.Label(toolbar, text="Ready", font=("Arial", 10, "italic"))
        self.status.pack(side=tk.LEFT, padx=10)

        self.viewport = ttk.Label(self.root)
        self.viewport.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.cam_btn.configure(text="Stop Webcam")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.cam_btn.configure(text="Start Webcam")

    def select_roi(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Open native OpenCV ROI selection window
                bbox = cv2.selectROI("Select Target & Press SPACE/ENTER", frame, fromCenter=False)
                cv2.destroyWindow("Select Target & Press SPACE/ENTER")

                if bbox != (0, 0, 0, 0):
                    # Initialise OpenCV legacy CSRT or KCF tracker
                    self.tracker = cv2.TrackerMIL_create()
                    self.tracker.init(frame, bbox)
                    self.is_tracking = True
                    self.status.configure(text="Tracking Active")

    def reset_tracker(self):
        self.is_tracking = False
        self.tracker = None
        self.status.configure(text="Tracker Reset")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            if self.is_tracking and self.tracker is not None:
                success, bbox = self.tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Tracking", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Target Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.viewport.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeTrackerApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()