import tkinter as tk
from tkinter import filedialog, ttk
import os
from PIL import Image, ImageTk
import torch
from transformers import CLIPProcessor, CLIPModel

class CLIPSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 74: AI Image Search Engine (CLIP)")
        self.root.geometry("900x700")

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.folder_path = ""

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Select Folder", command=self.select_folder).pack(side=tk.LEFT, padx=3)
        self.query_entry = ttk.Entry(toolbar, width=30)
        self.query_entry.insert(0, "a dog playing with a ball")
        self.query_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(toolbar, text="Search Images", command=self.search_images).pack(side=tk.LEFT, padx=3)

        self.lbl_result = ttk.Label(self.root, text="Match: None", font=("Arial", 11, "bold"))
        self.lbl_result.pack(pady=5)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def select_folder(self):
        self.folder_path = filedialog.askdirectory()

    def search_images(self):
        if not self.folder_path:
            return

        query = self.query_entry.get().strip()
        files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not files:
            return

        images = [Image.open(f).convert("RGB") for f in files]
        inputs = self.processor(text=[query], images=images, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=0)
            best_idx = probs.argmax().item()

        best_img_path = files[best_idx]
        self.lbl_result.configure(text=f"Best Match: {os.path.basename(best_img_path)}")

        best_img = Image.open(best_img_path)
        best_img.thumbnail((750, 550))
        self.tk_photo = ImageTk.PhotoImage(best_img)
        self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = CLIPSearchApp(root)
    root.mainloop()