import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

class ImageCaptioningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 72: AI Image Caption Generator")
        self.root.geometry("950x700")

        # Load HuggingFace Vision-Encoder-Decoder model
        self.model_name = "nlpconnect/vit-gpt2-image-captioning"
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image & Generate Caption", command=self.generate_caption).pack(side=tk.LEFT, padx=4)

        self.caption_lbl = ttk.Label(self.root, text="Caption: Load an image to describe.", 
                                     font=("Arial", 12, "italic"), wraplength=850)
        self.caption_lbl.pack(pady=10)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def generate_caption(self):
        path = filedialog.askopenfilename()
        if path:
            raw_image = Image.open(path).convert('RGB')
            pixel_values = self.processor(images=raw_image, return_tensors="pt").pixel_values.to(self.device)

            output_ids = self.model.generate(pixel_values, max_length=16, num_beams=4)
            preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            caption = preds[0].strip().capitalize()

            self.caption_lbl.configure(text=f'Caption: "{caption}."')

            raw_image.thumbnail((800, 550))
            self.tk_photo = ImageTk.PhotoImage(raw_image)
            self.display.configure(image=self.tk_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptioningApp(root)
    root.mainloop()