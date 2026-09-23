import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, rects):
        objects_bbs_ids = []
        for rect in rects:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for obj_id, pt in self.center_points.items():
                dist = np.hypot(cx - pt[0], cy - pt[1])
                if dist < 40:
                    self.center_points[obj_id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, obj_id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            new_center_points[object_id] = self.center_points[object_id]
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

class PeopleTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 32: People Tracking")
        self.root.geometry("900x700")

        self.cap = None
        self.is_running = False
        self.tracker = EuclideanDistTracker()
        self.detector = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Video", command=self.load_video).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Webcam", command=lambda: self.start_stream(0)).pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_video(self):
        path = filedialog.askopenfilename()
        if path:
            self.start_stream(path)

    def start_stream(self, src):
        self.stop_stream()
        self.cap = cv2.VideoCapture(src)
        if self.cap.isOpened():
            self.is_running = True
            self.loop()

    def stop_stream(self):
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            mask = self.detector.apply(frame)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            detections = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1200:
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append([x, y, w, h])

            boxes_ids = self.tracker.update(detections)
            for box_id in boxes_ids:
                x, y, w, h, obj_id = box_id
                cv2.putText(frame, f"ID {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.4, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((850, 600))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

            self.root.after(20, self.loop)
        else:
            self.stop_stream()

if __name__ == "__main__":
    root = tk.Tk()
    app = PeopleTrackingApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_stream(), root.destroy()))
    root.mainloop()