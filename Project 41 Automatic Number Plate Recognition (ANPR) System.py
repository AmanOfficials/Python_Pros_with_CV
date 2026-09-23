import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import sqlite3
import pytesseract
from datetime import datetime
from PIL import Image, ImageTk

class ANPRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 41: ANPR with SQLite Database")
        self.root.geometry("1100x700")

        self.cv_img = None
        self._init_db()

        # UI Layout: Left split for display, Right split for database logs
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(left_panel, padding=5)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text="Load Vehicle Image", command=self.load_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Scan & Register Plate", command=self.process_anpr).pack(side=tk.LEFT, padx=3)

        self.display = ttk.Label(left_panel)
        self.display.pack(expand=True, fill=tk.BOTH, pady=5)

        # Database Log Table
        right_panel = ttk.Frame(main_frame, width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

        ttk.Label(right_panel, text="Vehicle Access Logs", font=("Arial", 11, "bold")).pack(pady=5)
        self.tree = ttk.Treeview(right_panel, columns=("ID", "Plate", "Timestamp"), show="headings", height=20)
        self.tree.heading("ID", text="ID")
        self.tree.column("ID", width=40)
        self.tree.heading("Plate", text="Plate Number")
        self.tree.column("Plate", width=120)
        self.tree.heading("Timestamp", text="Timestamp")
        self.tree.column("Timestamp", width=160)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.refresh_table()

    def _init_db(self):
        self.conn = sqlite3.connect("anpr_records.db")
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS records (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            plate TEXT,
                            timestamp TEXT)''')
        self.conn.commit()

    def refresh_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM records ORDER BY id DESC")
        for row in cursor.fetchall():
            self.tree.insert("", tk.END, values=row)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            self.cv_img = cv2.imread(path)
            self.render(self.cv_img)

    def process_anpr(self):
        if self.cv_img is None:
            return

        img = self.cv_img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)

        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        plate_text = ""
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 2.0 <= aspect_ratio <= 6.0:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    roi = gray[y:y+h, x:x+w]
                    _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    cfg = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    plate_text = pytesseract.image_to_string(roi_thresh, config=cfg).strip()
                    break

        if plate_text:
            cursor = self.conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("INSERT INTO records (plate, timestamp) VALUES (?, ?)", (plate_text, timestamp))
            self.conn.commit()
            self.refresh_table()
            cv2.putText(img, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        self.render(img)

    def render(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((700, 550))
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = ANPRApp(root)
    root.mainloop()