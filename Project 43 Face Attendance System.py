import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import csv
from datetime import datetime
import numpy as np
from PIL import Image, ImageTk

class AttendanceSystemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 43: Automated Face Attendance System")
        self.root.geometry("900x650")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = None
        self.is_running = False
        self.marked_names = set()

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Attendance Camera", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        ttk.Button(toolbar, text="View CSV Attendance Sheet", command=self.show_records).pack(side=tk.LEFT, padx=4)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn.configure(text="Stop Attendance Camera")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn.configure(text="Start Attendance Camera")

    def mark_attendance(self, name):
        if name not in self.marked_names:
            now = datetime.now()
            time_str = now.strftime('%H:%M:%S')
            date_str = now.strftime('%Y-%m-%d')
            with open('attendance.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, date_str, time_str])
            self.marked_names.add(name)

    def show_records(self):
        try:
            with open('attendance.csv', 'r') as f:
                data = f.read()
            messagebox.showinfo("Attendance Records", data if data else "Empty sheet.")
        except FileNotFoundError:
            messagebox.showinfo("Attendance Records", "No attendance marked yet.")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                # Mock identity matching for verification pipeline
                person_name = "Employee_01"
                self.mark_attendance(person_name)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{person_name}: Logged", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((850, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystemApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()