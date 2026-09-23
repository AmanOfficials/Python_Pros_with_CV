import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

class VideoSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 54: Video Object Search")
        self.root.geometry("1000x700")

        self.model = YOLO("yolov8n.pt")
        self.video_path = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Select Video", command=self.pick_video).pack(side=tk.LEFT, padx=3)

        ttk.Label(toolbar, text="Find Target:").pack(side=tk.LEFT, padx=(10, 2))
        self.query_entry = ttk.Entry(toolbar, width=15)
        self.query_entry.insert(0, "bottle")
        self.query_entry.pack(side=tk.LEFT, padx=3)

        ttk.Button(toolbar, text="Run Index Search", command=self.search_video).pack(side=tk.LEFT, padx=3)

        # Main Layout: Canvas preview on left, matched timestamps on right
        content = ttk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.display = ttk.Label(content)
        self.display.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.match_list = tk.Listbox(content, width=25)
        self.match_list.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

    def pick_video(self):
        self.video_path = filedialog.askopenfilename()

    def search_video(self):
        if not self.video_path:
            return

        target_class = self.query_entry.get().strip().lower()
        self.match_list.delete(0, tk.END)

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process 1 frame per second to speed up video indexing
            if frame_idx % int(fps) == 0:
                results = self.model.predict(frame, verbose=False)
                for box in results[0].boxes:
                    cls_name = self.model.names[int(box.cls[0])].lower()
                    if target_class in cls_name:
                        sec = int(frame_idx / fps)
                        time_str = f"{sec // 60:02d}:{sec % 60:02d}"
                        self.match_list.insert(tk.END, f"Found at {time_str}")

                        # Preview match
                        annotated = results[0].plot()
                        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb)
                        pil_img.thumbnail((650, 500))
                        self.tk_photo = ImageTk.PhotoImage(pil_img)
                        self.display.configure(image=self.tk_photo)
                        self.root.update()
                        break

            frame_idx += 1
        cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoSearchApp(root)
    root.mainloop()