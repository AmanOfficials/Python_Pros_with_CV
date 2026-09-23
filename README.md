# 👁️ 100 Computer Vision Projects with Python

> A structured, hands-on collection of **100 Computer Vision projects** built with Python, OpenCV, Tkinter, MediaPipe, YOLO, PyTorch, OCR, and modern AI/Deep Learning techniques.

This repository is designed as a practical **Computer Vision learning roadmap and portfolio**, starting from fundamental image processing and gradually progressing toward real-time detection, tracking, segmentation, deep learning, AI-powered vision systems, and advanced computer vision research concepts.

The projects are organized from **Beginner → Intermediate → Advanced → Deep Learning → AI → Expert Level**, allowing learners to build their Computer Vision skills progressively through practical applications.

---

## 🚀 What You Will Learn

By completing these projects, you will gain practical experience with:

- Python for Computer Vision
- OpenCV
- NumPy
- PIL / Pillow
- Tkinter GUI development
- Image processing
- Image filtering
- Edge detection
- Thresholding
- Contours
- Object detection
- Object tracking
- Face detection
- Face recognition
- Hand tracking
- Gesture recognition
- Computer vision geometry
- OCR
- MediaPipe
- YOLO
- ByteTrack / DeepSORT
- PyTorch
- CNNs
- Image classification
- Semantic segmentation
- Instance segmentation
- Pose estimation
- Depth estimation
- Stereo vision
- Visual SLAM
- Image enhancement
- AI-powered image and video analysis

---

# 📚 Project Roadmap

The repository contains **100 projects** divided into six progressive levels.

| Level | Projects | Focus |
|---|---:|---|
| 🟢 Beginner | 1–15 | OpenCV & Image Processing |
| 🟡 Intermediate | 16–35 | Detection & Tracking |
| 🟠 Advanced | 36–55 | Real-Time Computer Vision |
| 🔵 Deep Learning | 56–70 | CNNs & Neural Networks |
| 🟣 AI + Computer Vision | 71–85 | AI Vision Applications |
| 🔴 Expert | 86–100 | Advanced Vision Systems |

---

# 🟢 Level 1 — Beginner: OpenCV Basics

These projects introduce the fundamental building blocks of Computer Vision.

### 01. Image Viewer & Editor
Resize, crop, rotate, flip, adjust brightness, and modify image contrast.

### 02. Grayscale Image Converter
Convert RGB/BGR images into grayscale representations.

### 03. Image Filtering App
Experiment with Gaussian, Median, Bilateral, and Box filters.

### 04. Edge Detection Tool
Implement Sobel, Laplacian, and Canny edge detection.

### 05. Image Thresholding App
Explore Binary, Adaptive, and Otsu thresholding techniques.

### 06. Color Detection System
Detect specific colors using HSV masking.

### 07. Object Counter
Count objects using contour properties.

### 08. Shape Detection
Identify circles, triangles, rectangles, and squares.

### 09. Contour Detection App
Find, filter, and visualize external object boundaries.

### 10. Image Histogram Analyzer
Display RGB and grayscale pixel distributions.

### 11. Document Scanner
Detect document corners and perform four-point perspective transformation.

### 12. Image Blur & Sharpening Tool
Compare different blur and sharpening techniques.

### 13. Motion Detection
Detect movement using frame differencing and background subtraction.

### 14. Webcam Color Tracker
Track selected colors in real time using a webcam.

### 15. Virtual Drawing Board
Create an air-canvas drawing application using a colored object as a virtual pen.

These first projects focus on the core OpenCV operations used throughout the rest of the repository.

---

# 🟡 Level 2 — Intermediate: Detection & Tracking

Projects at this level introduce real-time detection, tracking, facial landmarks, and human interaction.

### 16. Face Detection System
Detect faces using Haar Cascades or OpenCV DNN.

### 17. Eye Detection System
Perform real-time eye detection.

### 18. Face Counting System
Count visible faces through a webcam.

### 19. People Counter
Count people entering and leaving through a virtual line.

### 20. Real-Time Object Tracker
Track objects using CSRT, KCF, or MOSSE trackers.

### 21. Vehicle Counter
Detect vehicles and count them across a defined region.

### 22. Traffic Monitoring System
Analyze vehicle classifications and traffic density.

### 23. License Plate Detection
Locate license plate regions in images or video.

### 24. License Plate Character Recognition
Extract license plates and recognize their text using OCR.

### 25. Hand Detection
Detect hands using MediaPipe or OpenCV.

### 26. Finger Counter
Count raised fingers using hand landmark geometry.

### 27. Hand Gesture Controller
Control system functions such as volume, brightness, or slides using gestures.

### 28. Virtual Mouse
Control the computer pointer using fingertip positions.

### 29. Virtual Keyboard
Operate an on-screen keyboard using finger movements.

### 30. Face Landmark Detection
Visualize facial landmark points or facial mesh.

### 31. Head Pose Estimation
Estimate facial pitch, yaw, and roll.

### 32. People Tracking
Track multiple people across sequential video frames.

### 33. Object Tracking Dashboard
Display tracking IDs, movement paths, and speed information.

### 34. Automatic Background Removal
Remove image backgrounds using GrabCut or contour-based segmentation.

### 35. Document Perspective Correction
Automatically detect and align warped documents.



---

# 🟠 Level 3 — Advanced Computer Vision

These projects introduce modern object detection, tracking, analytics, segmentation, and vision-based systems.

### 36. Real-Time Object Detection
Build multi-class object detection using YOLO.

### 37. Custom Object Detection
Train and detect custom object classes.

### 38. Object Detection + Tracking
Combine YOLO with ByteTrack or DeepSORT.

### 39. Smart Traffic Camera
Build a traffic monitoring pipeline with detection, tracking, and counting.

### 40. Parking Space Detection
Detect whether parking spaces are free or occupied.

### 41. Automatic Number Plate Recognition — ANPR
Create an end-to-end number plate detection, OCR, and SQLite logging system.

### 42. Face Recognition System
Match detected faces against an enrolled database.

### 43. Face Recognition Attendance System
Record attendance automatically with timestamps.

### 44. People Counting & Analytics
Generate hourly footfall graphs and occupancy alerts.

### 45. Human Pose Estimation
Detect human skeletal landmarks.

### 46. Exercise / Pose Recognition
Build automatic squat and push-up repetition counters.

### 47. Gesture Recognition System
Classify dynamic hand gestures.

### 48. Real-Time Segmentation
Extract foreground masks from webcam streams.

### 49. Road Lane Detection
Detect road lanes using edges, ROI masking, and Hough transforms.

### 50. Road Sign Detection
Identify and classify traffic signs.

### 51. Vehicle Speed Estimation
Estimate vehicle speed using frame-rate and pixel displacement.

### 52. Accident Detection Prototype
Detect potential accidents using motion and vehicle overlap patterns.

### 53. Crowd Analysis
Generate crowd-density maps and heatmaps.

### 54. Video Object Search
Index video frames and search for specific visual targets.

### 55. Image Similarity System
Compare images using SIFT, ORB, or embedding-based similarity.



---

# 🔵 Level 4 — Deep Learning Computer Vision

These projects introduce neural networks and deep-learning-based visual recognition.

### 56. CNN Image Classifier
Build a custom CNN for multi-class image classification.

### 57. Animal Image Classifier
Classify different animal species.

### 58. Plant Disease Classifier
Classify plant diseases from leaf images.

### 59. Waste Classification
Classify recyclable materials such as paper, plastic, metal, and glass.

### 60. Food Image Classification
Classify food images and optionally estimate food categories.

### 61. Fashion Image Classifier
Classify clothing and footwear.

### 62. Handwritten Digit Recognition
Recognize handwritten digits using the MNIST dataset.

### 63. Handwritten Character Recognition
Recognize handwritten alphabets and symbols.

### 64. Emotion Classification
Perform facial emotion classification.

### 65. Age Group Classification
Estimate age categories from facial images.

### 66. Scene Classification
Classify environments such as indoor and outdoor scenes.

### 67. U-Net Image Segmentation
Perform pixel-level segmentation using U-Net.

### 68. Semantic Segmentation
Classify every image pixel using models such as DeepLabV3 or PSPNet.

### 69. Instance Segmentation
Detect individual object instances using Mask R-CNN or YOLO-Seg.

### 70. Neural Network Pose Classification
Classify poses using landmark coordinates and neural networks.



---

# 🟣 Level 5 — AI + Computer Vision

These projects combine Computer Vision with AI, OCR, embeddings, multimodal models, and intelligent analytics.

### 71. AI Object Detection Assistant
Detect objects and convert visual information into speech.

### 72. AI Image Caption Generator
Generate natural-language descriptions of images using models such as ViT-GPT2 or BLIP.

### 73. Visual Question Answering — VQA
Ask questions about an image and generate visual answers.

### 74. AI Image Search Engine
Search visually similar images using CLIP embeddings and vector search.

### 75. Smart Surveillance Prototype
Detect unauthorized presence and generate alerts.

### 76. AI Document Understanding
Extract structured information from documents using LayoutLM-style models.

### 77. AI OCR System
Extract multilingual text using PaddleOCR or Tesseract.

### 78. Handwritten Text Recognition
Convert handwritten sentences and lines into machine-readable text.

### 79. Visual Product Search
Find visually similar products using image embeddings.

### 80. AI Image Quality Analyzer
Analyze blur, noise, and lighting quality.

### 81. Image Forgery Detection
Explore Error Level Analysis and image-splicing detection.

### 82. AI Background Segmentation
Use deep learning for intelligent background replacement.

### 83. Real-Time Scene Understanding
Combine object detection, scene text, and depth information.

### 84. Visual Navigation Prototype
Generate obstacle-avoidance alerts for robotics or accessibility applications.

### 85. AI-Powered Video Analytics Dashboard
Create a dashboard for multi-camera detection and analytics.



---

# 🔴 Level 6 — Expert Computer Vision Projects

The final projects focus on advanced computer vision research and production-style systems.

### 86. YOLO Custom Object Detection Platform
Create an interface for dataset annotation, training, and custom YOLO detection.

### 87. Multi-Object Tracking System
Implement multi-camera tracking with person/object re-identification.

### 88. Real-Time Semantic Segmentation
Build a high-FPS semantic segmentation pipeline.

### 89. Real-Time Instance Segmentation
Generate real-time object instance masks.

### 90. 3D Object Detection
Explore point-cloud or monocular 3D bounding-box prediction.

### 91. Stereo Vision Depth Estimation
Calculate depth using disparity between stereo camera images.

### 92. Monocular Depth Estimation
Generate depth maps from a single camera using models such as MiDaS.

### 93. Visual SLAM Prototype
Build camera-based mapping and self-localization.

### 94. Image Super-Resolution
Upscale low-resolution images using Real-ESRGAN or SRCNN.

### 95. Low-Light Image Enhancement
Improve poorly illuminated images using Retinex or EnlightenGAN approaches.

### 96. Image Denoising Neural Network
Build an autoencoder-based image noise removal system.

### 97. Image Deblurring Neural Network
Recover blurred images using neural-network-based deblurring.

### 98. Neural Style Transfer
Transfer artistic textures and visual styles between images.

### 99. Generative Image Restoration
Restore missing image regions using inpainting, GANs, or diffusion techniques.

### 100. End-to-End Computer Vision Analytics Platform
Build a complete industrial-style platform containing vision pipelines, alerts, monitoring, and analytics dashboards.



---

# 🏗️ Universal Project Architecture

Most desktop-based projects in this repository can follow a modular **Tkinter + OpenCV architecture**.

The basic architecture consists of:

```text
Computer Vision Application
│
├── GUI Layer
│   └── Tkinter / ttk
│
├── Input Layer
│   ├── Image
│   ├── Video
│   └── Webcam
│
├── Processing Layer
│   ├── OpenCV
│   ├── NumPy
│   ├── MediaPipe
│   └── AI / Deep Learning Models
│
├── Visualization Layer
│   └── Tkinter / PIL
│
└── Output Layer
    ├── Images
    ├── Video
    ├── CSV
    ├── SQLite
    └── Analytics
```

A reusable Tkinter/OpenCV framework can provide image loading, webcam streaming, algorithm selection, image rendering, and output saving.

---

# 🛠️ Technology Stack

## Core Technologies

- Python 3
- OpenCV
- NumPy
- Pillow
- Tkinter
- Matplotlib

## Computer Vision

- OpenCV
- OpenCV Contrib
- MediaPipe
- Haar Cascades
- OpenCV DNN

## Object Detection

- YOLO
- Ultralytics
- ByteTrack
- DeepSORT

## Deep Learning

- PyTorch
- TorchVision
- CNN
- U-Net
- DeepLabV3
- Mask R-CNN

## OCR

- Tesseract
- Pytesseract
- PaddleOCR

## AI / Vision Models

- CLIP
- BLIP
- ViT
- LayoutLM
- MiDaS
- Real-ESRGAN

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/100-computer-vision-projects.git
```

```bash
cd 100-computer-vision-projects
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Core Dependencies

```bash
pip install opencv-python opencv-contrib-python pillow numpy matplotlib
```

## 4. Install Advanced Dependencies

```bash
pip install mediapipe ultralytics torch torchvision torchaudio pytesseract
```

The project source recommends the core OpenCV/Pillow/NumPy/Matplotlib stack first, followed by MediaPipe, Ultralytics, PyTorch, TorchVision, TorchAudio, and Pytesseract as the projects become more advanced.

---

# ▶️ Running a Project

Each project can be organized as an independent application:

```bash
cd projects/project-01-image-viewer
python main.py
```

For webcam-based projects, make sure your camera is available and that the application has permission to access it.

---

# 📁 Recommended Repository Structure

```text
100-computer-vision-projects/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── beginner/
│   ├── 01-image-viewer/
│   ├── 02-grayscale/
│   ├── 03-image-filtering/
│   ├── 04-edge-detection/
│   └── ...
│
├── intermediate/
│   ├── 16-face-detection/
│   ├── 17-eye-detection/
│   ├── 18-face-counting/
│   └── ...
│
├── advanced/
│   ├── 36-yolo-detection/
│   ├── 37-custom-detection/
│   ├── 38-object-tracking/
│   └── ...
│
├── deep-learning/
│   ├── 56-cnn-classifier/
│   ├── 57-animal-classifier/
│   └── ...
│
├── ai-computer-vision/
│   ├── 71-ai-object-assistant/
│   ├── 72-image-captioning/
│   └── ...
│
└── expert/
    ├── 86-yolo-platform/
    ├── 87-multi-object-tracking/
    └── ...
```

---

# 📈 Suggested Learning Path

Do not attempt to build all 100 projects simultaneously.

Follow this progression:

```text
Python
   ↓
NumPy
   ↓
OpenCV Basics
   ↓
Image Processing
   ↓
Contours & Geometry
   ↓
Face / Object Detection
   ↓
Tracking
   ↓
MediaPipe
   ↓
YOLO
   ↓
Deep Learning
   ↓
CNNs
   ↓
Segmentation
   ↓
OCR
   ↓
Vision + AI
   ↓
Advanced Computer Vision
```

The source roadmap similarly recommends progressing from basic OpenCV image processing through detection/tracking, advanced real-time systems, deep learning, and finally AI/expert systems.

---

# 🎯 Project Development Strategy

For every project, try to implement the following stages:

### Step 1 — Understand

Learn the Computer Vision concept behind the project.

### Step 2 — Build the Basic Version

Create a simple Python/OpenCV implementation.

### Step 3 — Add GUI

Create a Tkinter interface.

### Step 4 — Add Real-Time Processing

Connect webcam/video input where appropriate.

### Step 5 — Add Configuration

Allow users to change thresholds, modes, confidence levels, or other parameters.

### Step 6 — Add Output

Save images, videos, logs, CSV files, or database records.

### Step 7 — Improve the Project

Add error handling, better UI, performance optimization, and documentation.

### Step 8 — Document

Every project should contain:

```text
README.md
requirements.txt
main.py
screenshots/
sample_data/
models/
```

---

# 📊 Project Status

| Category | Projects | Status |
|---|---:|---|
| OpenCV Basics | 15 | ⬜ |
| Detection & Tracking | 20 | ⬜ |
| Advanced CV | 20 | ⬜ |
| Deep Learning | 15 | ⬜ |
| AI + Computer Vision | 15 | ⬜ |
| Expert Systems | 15 | ⬜ |
| **Total** | **100** | **⬜** |

Update the status as you complete each project:

```text
⬜ Not Started
🟡 In Progress
✅ Completed
```

---

# 🧠 What Makes This Repository Different?

This is not intended to be a collection of isolated code snippets.

The goal is to build a **progressive Computer Vision portfolio** where each project introduces a new concept and prepares you for the next level.

You move from:

**Pixels → Image Processing → Detection → Tracking → Deep Learning → AI → Advanced Vision Systems**

---

# 💼 Portfolio Value

These projects can demonstrate practical skills in:

- Computer Vision
- Python Development
- AI / Machine Learning
- Deep Learning
- Image Processing
- Video Analytics
- Object Detection
- Object Tracking
- OCR
- Desktop GUI Development
- AI Application Development
- Real-Time Systems

For a professional portfolio, prioritize projects that demonstrate an end-to-end workflow rather than simply showing a model prediction.

---

# 🔬 Advanced Concepts

The later projects explore advanced areas including:

- Multi-object tracking
- Re-identification
- 3D computer vision
- Stereo depth
- Monocular depth
- Visual SLAM
- Super-resolution
- Image restoration
- Image enhancement
- Neural style transfer
- Generative restoration
- Industrial computer vision analytics

---

# ⚡ Performance Considerations

Real-time Computer Vision applications can be computationally expensive.

For better performance:

- Resize frames before processing.
- Avoid unnecessary image conversions.
- Process every Nth frame when appropriate.
- Use GPU acceleration when available.
- Use lightweight models for real-time applications.
- Separate GUI and processing responsibilities.
- Avoid blocking the Tkinter event loop.
- Release webcam/video resources correctly.

For webcam applications, the architecture can use Tkinter's event loop scheduling to continuously process frames without blocking the GUI.

---

# 🔐 Privacy & Responsible Use

Some projects involve sensitive technologies such as:

- Face recognition
- Face detection
- Surveillance
- License plate recognition
- Attendance systems
- Human tracking

Use these technologies responsibly.

When deploying real-world systems, consider:

- User consent
- Data protection
- Secure storage
- Access control
- Applicable laws and regulations
- False-positive handling
- Bias and model limitations

This repository is intended primarily for **education, experimentation, and portfolio development**.

---

# 🤝 Contributing

Contributions are welcome.

You can contribute by:

1. Forking the repository.
2. Creating a feature branch.
3. Adding or improving a project.
4. Adding documentation.
5. Improving performance.
6. Adding tests.
7. Submitting a Pull Request.

Example:

```bash
git checkout -b feature/new-project
```

```bash
git add .
git commit -m "Add new computer vision project"
git push origin feature/new-project
```

Then open a Pull Request.

---

# 📝 Project Documentation Template

Each project should ideally contain:

```markdown
# Project Name

## Overview

Short explanation of the project.

## Features

- Feature 1
- Feature 2
- Feature 3

## Technologies

- Python
- OpenCV
- NumPy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## How It Works

Explain the Computer Vision pipeline.

## Screenshots

Add screenshots here.

## Future Improvements

List possible improvements.

## License

See the repository license.
```

---

# 📚 Learning Resources

Recommended topics to study alongside the projects:

- Python programming
- NumPy
- Linear algebra
- Image processing
- OpenCV
- Machine Learning
- Deep Learning
- CNN architectures
- Object detection
- Image segmentation
- Computer Vision mathematics

---

# ⭐ Support the Project

If you find this repository useful:

⭐ Star the repository  
🍴 Fork the repository  
🐛 Report issues  
💡 Suggest new projects  
🤝 Contribute improvements  

---

# 👨‍💻 Author

**Amanullah Panhwar**

Computer Science & Technology Enthusiast  
Python • Android • Computer Vision • AI • Machine Learning

---

# 📜 License

This project is intended for educational and portfolio purposes.

Add your preferred open-source license to the repository before distributing the project publicly.

---

## 🚀 Start Building

The best way to learn Computer Vision is to **build, experiment, break things, debug them, and improve them**.

Start with Project 01 and progressively work toward Project 100.

```text
01 → 15 → 35 → 55 → 70 → 85 → 100

OpenCV
  ↓
Detection
  ↓
Tracking
  ↓
Deep Learning
  ↓
AI Vision
  ↓
Expert Computer Vision
```

**100 Projects. One Computer Vision Journey.**