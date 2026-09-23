import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

class TrackingDashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 33: Object Tracking Dashboard")
        self.root.geometry("1050x700")

        self.cap = None
        self.is_running = False
        self.tracker = None
        self.is_tracking = False

        # Build Sidebar Dashboard Table
        self.side_panel = ttk.Frame(self.root, padding=10, width=280)
        self.side_panel.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(self.side_panel, text="Telemetry Dashboard", font=("Arial", 12, "bold")).pack(pady=5)
        self.lbl_pos = ttk.Label(self.side_panel, text="Position (X, Y): N/A")
        self.lbl_pos.pack(anchor=tk.W, pady=3)
        self.lbl_size = ttk.Label(self.side_panel, text="Size (W, H): N/A")
        self.lbl_size.pack(anchor=tk.W, pady=3)
        self.lbl_speed = ttk.Label(self.side_panel, text="Displacement: 0 px/frame")
        self.lbl_speed.pack(anchor=tk.W, pady=3)

        self.prev_coord = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Start Webcam", command=self.toggle_stream).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Select Track ROI", command=self.select_roi).pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()

    def select_roi(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                bbox = cv2.selectROI("Dashboard Selection", frame, False)
                cv2.destroyWindow("Dashboard Selection")
                if bbox != (0, 0, 0, 0):
                    self.tracker = cv2.TrackerMIL_create()
                    self.tracker.init(frame, bbox)
                    self.is_tracking = True

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            if self.is_tracking and self.tracker:
                success, bbox = self.tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    cx, cy = x + w // 2, y + h // 2
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                    # Calculate velocity/displacement
                    disp = 0
                    if self.prev_coord:
                        disp = int(np.hypot(cx - self.prev_coord[0], cy - self.prev_coord[1]))
                    self.prev_coord = (cx, cy)

                    self.lbl_pos.configure(text=f"Position (X, Y): ({cx}, {cy})")
                    self.lbl_size.configure(text=f"Size (W, H): {w} x {h}")
                    self.lbl_speed.configure(text=f"Displacement: {disp} px/frame")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((750, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = TrackingDashboardApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()