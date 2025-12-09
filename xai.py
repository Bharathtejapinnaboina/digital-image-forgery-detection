import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

def preprocess_image(image_path, device):
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),  # Scales to [0, 1]
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    return img_tensor, img

def get_saliency_map(model, img_tensor, target_idx):
    img_tensor = img_tensor.clone().requires_grad_(True)
    
    # Run inference
    results = model(img_tensor)
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return None, "No detections for saliency map"
    
    # Switch to training mode for gradients
    model.model.train()
    with torch.enable_grad():
        preds = model.model.forward(img_tensor)
        pred = preds[0]  # Shape: [1, 65, 80, 80] (largest scale)
        pred = pred.clone().requires_grad_(True)
        
        # Get bounding box for target_idx
        boxes = results[0].boxes
        xywh = boxes.xywh[target_idx].cpu().numpy()
        x_center, y_center = xywh[0], xywh[1]
        
        # Map to 80x80 grid (stride 8 for largest scale)
        scale_factor = 640 / 80  # Input size / grid size
        x, y = int(x_center / scale_factor), int(y_center / scale_factor)
        x, y = min(max(x, 0), 79), min(max(y, 0), 79)  # Clamp
        
        # Debug location and confidence
        target_conf = pred[0, 4, y, x]  # Objectness score
        print(f"Target idx: {target_idx}, (x, y): ({x}, {y}), Target conf: {target_conf.item()}")
        
        # Backpropagate
        model.zero_grad()
        target_conf.backward()
        
        # Debug gradients
        print(f"Max gradient: {img_tensor.grad.max().item()}")
    
    # Reset model to eval mode
    model.model.eval()
    
    # Compute saliency
    saliency = img_tensor.grad.data.abs().squeeze().cpu()
    saliency = saliency.permute(1, 2, 0).numpy()
    saliency_map = np.maximum(saliency, 0)
    saliency_map = saliency_map / (saliency_map.max() + 1e-8)
    saliency_map = np.uint8(saliency_map * 255)
    return saliency_map, None

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("best1l.pt")
image_path = 'D:/Forgery Detection Final/Forgery Detection Final/images/image-2-_png_jpg.rf.952fa2179ee36dc73b462f225d568f27.jpg'
img_tensor, img = preprocess_image(image_path, device)

# Inference
results = model(img_tensor)
img_np = np.array(img)

if results[0].boxes is not None and len(results[0].boxes) > 0:
    boxes = results[0].boxes
    xywh = boxes.xywh.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    names = results[0].names

    confidence_threshold = 0.2
    valid_boxes = conf >= confidence_threshold

    if valid_boxes.sum() > 0:
        for idx in range(len(valid_boxes)):
            if valid_boxes[idx]:
                # Draw bounding box
                x1, y1, w, h = xywh[idx]
                x1, y1 = int(x1 - w / 2), int(y1 - h / 2)
                x2, y2 = int(x1 + w), int(y1 + h)
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{names[int(cls[idx])]}: {conf[idx]:.2f}'
                cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Generate saliency map
                saliency_map, error = get_saliency_map(model, img_tensor, idx)
                if saliency_map is not None:
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    plt.imshow(img_np)
                    plt.axis('off')
                    plt.title(f"Forgery Prediction")

                    plt.subplot(1, 2, 2)
                    plt.imshow(saliency_map, cmap='hot')
                    plt.axis('off')
                    plt.title(f"Saliency Map")
                    plt.show()
                else:
                    print(error)
    else:
        print("No valid detections above confidence threshold.")
else:
    print("No objects detected in the image.")