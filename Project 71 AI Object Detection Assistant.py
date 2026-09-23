import tkinter as tk
from tkinter import ttk
import cv2
import pyttsx3
import threading
from PIL import Image, ImageTk
from ultralytics import YOLO

class VoiceAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 71: AI Object Detection Assistant")
        self.root.geometry("850x650")

        self.model = YOLO("yolov8n.pt")
        self.cap = None
        self.is_running = False
        self.spoken_objects = set()

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn = ttk.Button(toolbar, text="Start Assistant", command=self.toggle_stream)
        self.btn.pack(side=tk.LEFT, padx=4)

        self.lbl_status = ttk.Label(toolbar, text="Assistant: Idle", font=("Arial", 11, "bold"))
        self.lbl_status.pack(side=tk.LEFT, padx=15)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def speak(self, text):
        def _tts():
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
        threading.Thread(target=_tts, daemon=True).start()

    def toggle_stream(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.btn.configure(text="Stop Assistant")
                self.speak("Visual assistant online")
                self.loop()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn.configure(text="Start Assistant")

    def loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            results = self.model.predict(frame, conf=0.5, verbose=False)
            current_detected = set()

            for box in results[0].boxes:
                cls_name = self.model.names[int(box.cls[0])]
                current_detected.add(cls_name)

            # Speak new entities not recently voiced
            new_entities = current_detected - self.spoken_objects
            if new_entities:
                speech_text = f"I see a {', '.join(new_entities)}"
                self.lbl_status.configure(text=speech_text)
                self.speak(speech_text)
                self.spoken_objects = current_detected

            annotated = results[0].plot()
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

        self.root.after(30, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceAssistantApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, 'is_running', False), root.destroy()))
    root.mainloop()