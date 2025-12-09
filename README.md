# Forgery Detection & Localization in Digital Images  
Using YOLOv8, Faster R-CNN & RT-DETR

This project detects manipulated/forged regions in digital images using three powerful deep learning models:
- **YOLOv8**
- **Faster R-CNN**
- **RT-DETR (Transformer-based detector)**

## 🚀 Features
- Dataset preparation & preprocessing
- Training scripts for all three models
- Evaluation metrics: mAP, IoU
- Visualization of forged region detections
- Model comparison and result analysis

## 🧠 Models Used
### 1️⃣ YOLOv8
Fast & accurate one-stage detector.

### 2️⃣ Faster R-CNN
Two-stage detector with high precision.

### 3️⃣ RT-DETR
Real-time DETR (transformer-based detection system).

## 🧰 Tech Stack
- Python  
- PyTorch  
- Ultralytics YOLO  
- OpenCV  
- NumPy / Pandas  
- Matplotlib  

## 📂 Repository Structure
digital-image-forgery-detection/
├── dataset/
├── yolo/
├── faster_rcnn/
├── rtdetr/
├── results/
└── notebooks/

## ▶️ How to Run
1. Install dependencies:

pip install -r requirements.txt

2. Prepare dataset inside `dataset/` folder.  
3. Train YOLOv8:

python yolo/train_yolo.py

4. Train Faster R-CNN:

python faster_rcnn/train_faster_rcnn.py

5. Train RT-DETR:

python rtdetr/train_rtdetr.py

6. View detection samples in the `results/` folder.

## 📝 Notes
- Dataset is not included due to size.  
- Only sample images are provided (optional).  
- You can replace with your own manipulated image dataset.
