import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from transformers import pipeline

class VisualQuestionAnsweringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 73: Visual Question Answering")
        self.root.geometry("950x700")

        # Initialise VQA Pipeline
        self.vqa_pipeline = pipeline("visual-question-answering")
        self.img_path = None

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)

        ttk.Label(toolbar, text="Question:").pack(side=tk.LEFT, padx=(10, 2))
        self.q_entry = ttk.Entry(toolbar, width=35)
        self.q_entry.insert(0, "What is in the image?")
        self.q_entry.pack(side=tk.LEFT, padx=4)

        ttk.Button(toolbar, text="Ask Model", command=self.answer_question).pack(side=tk.LEFT, padx=4)

        self.ans_lbl = ttk.Label(self.root, text="Answer: N/A", font=("Arial", 12, "bold"))
        self.ans_lbl.pack(pady=8)

        self.display = ttk.Label(self.root)
        self.display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img_path = path
            pil_img = Image.open(path).convert('RGB')
            pil_img.thumbnail((750, 500))
            self.tk_photo = ImageTk.PhotoImage(pil_img)
            self.display.configure(image=self.tk_photo)

    def answer_question(self):
        if not self.img_path:
            return

        question = self.q_entry.get().strip()
        image = Image.open(self.img_path).convert('RGB')
        result = self.vqa_pipeline(image, question, top_k=1)
        answer = result[0]['answer']
        score = result[0]['score']

        self.ans_lbl.configure(text=f"Answer: {answer} (Confidence: {score*100:.1f}%)")

if __name__ == "__main__":
    root = tk.Tk()
    app = VisualQuestionAnsweringApp(root)
    root.mainloop()