import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import torch
import torch.nn as nn

# Lightweight MNIST CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

class DigitRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 62: Handwritten Digit Recognition")
        self.root.geometry("600x650")

        self.model = SimpleCNN()
        self.model.eval()

        self.canvas_size = 280
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack(pady=15)
        self.canvas.bind("<B1-Motion>", self.draw)

        # Offscreen PIL image buffer to track strokes
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw_buffer = ImageDraw.Draw(self.image)

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Recognize Digit", command=self.predict_digit).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear Canvas", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)

        self.pred_lbl = ttk.Label(self.root, text="Prediction: Draw a digit (0-9)", font=("Arial", 14, "bold"))
        self.pred_lbl.pack(pady=15)

    def draw(self, event):
        r = 10
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="white", outline="white")
        self.draw_buffer.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw_buffer = ImageDraw.Draw(self.image)
        self.pred_lbl.configure(text="Prediction: Draw a digit (0-9)")

    def predict_digit(self):
        # Resize canvas stroke buffer down to 28x28 MNIST dimension
        img_resized = self.image.resize((28, 28), Image.Resampling.BILINEAR)
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0
        tensor = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor)
            pred = torch.argmax(output, dim=1).item()

        self.pred_lbl.configure(text=f"Prediction: {pred}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    root.mainloop()