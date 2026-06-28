import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import argparse
import sys
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor

# ==========================================
# Configuration
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YOLO_MODEL_PATH = os.path.join(ROOT_DIR, "models", "weights", "yolo_v12m_best.pt")
UNET_MODEL_PATH = os.path.join(ROOT_DIR, "models", "weights", "unet_mit_b0_best.pt")
UNET_CSV = os.path.join(ROOT_DIR, "data", "fish_frames_unet.csv")

# Tape Calibration Constants (px to cm)
TOP_PX_PER_CM = 60.36 / 10.0
FRONT_PX_PER_CM = 65.35 / 10.0

feature_cols = [
    "Length (cm)", "Width (cm)", "Height (cm)",
    "Area (cm²)", "Perimeter (cm)",
    "Volume (cm³)", "Surface Area (cm²)",
    "Aspect Ratio", "Elongation", "Compactness",
    "Rectangularity", "Equivalent Diameter (cm)"
]

# ==========================================
# Model Loading & Training
# ==========================================
def get_trained_regression_model():
    print("[1/3] Loading dataset and training Regression Engine...")
    if not os.path.exists(UNET_CSV):
        print(f"Error: Could not find {UNET_CSV} to train the model!")
        sys.exit(1)
        
    df = pd.read_csv(UNET_CSV)
    df = df.dropna(subset=feature_cols + ["Weight (g)"])
    
    X = df[feature_cols].values
    y = df["Weight (g)"].values
    
    # Model 1: "LOFO" Deployment (Trained on 100% of the data)
    lofo_model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    lofo_model.fit(X, y)
    
    # Model 2: "Train-Test Split" Deployment (Trained on 80% of the data)
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    split_model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    split_model.fit(X_train, y_train)
    
    print(f"      -> Successfully trained LOFO model on {len(X)} frames.")
    print(f"      -> Successfully trained Split model on {len(X_train)} frames.")
    return lofo_model, split_model

def load_vision_models():
    print("[2/3] Loading Computer Vision Models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load YOLO
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    # Load UNet
    try:
        unet_model = smp.Unet(
            encoder_name="mit_b0",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        unet_model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=device))
    except:
        unet_model = smp.Unet(
            encoder_name="efficientnet-b3",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        unet_model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=device))
        
    unet_model.to(device)
    unet_model.eval()
    
    print(f"      -> Loaded YOLO and UNet successfully on {device}.")
    return yolo_model, unet_model, device

# ==========================================
# Image Processing
# ==========================================
def detect_all_fish(img_path, yolo_model):
    frame = cv2.imread(img_path)
    if frame is None: return None, []
    res = yolo_model(frame, verbose=False, imgsz=640)
    boxes = res[0].boxes.xyxy.cpu().numpy()
    confs = res[0].boxes.conf.cpu().numpy()
    
    # Filter out low confidence
    valid_indices = np.where(confs > 0.5)[0]
    boxes = boxes[valid_indices]
    confs = confs[valid_indices]
    
    # Sort boxes from left to right based on center X coordinate
    # This helps pair fish in Top view with fish in Front view
    if len(boxes) > 0:
        centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
        sort_idx = np.argsort(centers_x)
        boxes = boxes[sort_idx]
        
    return frame, boxes

def get_unet_mask(frame, unet_model, device):
    h, w = frame.shape[:2]
    transform = A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    augmented = transform(image=frame_rgb)
    tensor_img = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = unet_model(tensor_img)
        preds = (torch.sigmoid(logits) > 0.5).float().squeeze().cpu().numpy()
        
    mask_resized = cv2.resize(preds, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask_resized * 255).astype(np.uint8)

def process_fish_crop(frame, raw_mask, box, view="top"):
    # Clip mask to box
    x1, y1, x2, y2 = [int(v) for v in box]
    clipped = np.zeros_like(raw_mask)
    clipped[y1:y2, x1:x2] = raw_mask[y1:y2, x1:x2]
    
    # Remove fins
    k_size = max(3, int(min(x2-x1, y2-y1) * 0.15))
    if k_size % 2 == 0: k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    clean = cv2.morphologyEx(clipped, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, clean
    
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    
    if view == "top":
        rect = cv2.minAreaRect(c)
        (x, y), (w, h), angle = rect
        length = max(w, h)
        width = min(w, h)
    else:
        x, y, w, h = cv2.boundingRect(c)
        length = max(w, h)
        width = min(w, h)
        
    features = {
        "length_px": length, 
        "width_px": width, 
        "area_px": area, 
        "perimeter_px": perimeter, 
        "mask_px": np.sum(clean > 0)
    }
    
    return features, clean

# ==========================================
# Prediction Math
# ==========================================
def calculate_geometry(top_feat, front_feat):
    # Convert pixels to CM
    L_top = top_feat["length_px"] / TOP_PX_PER_CM
    W_top = top_feat["width_px"] / TOP_PX_PER_CM
    L_front = front_feat["length_px"] / FRONT_PX_PER_CM
    H_front = front_feat["width_px"] / FRONT_PX_PER_CM
    
    Area_top = top_feat["area_px"] / (TOP_PX_PER_CM**2)
    Area_front = front_feat["area_px"] / (FRONT_PX_PER_CM**2)
    Perim_top = top_feat["perimeter_px"] / TOP_PX_PER_CM
    
    # Averages
    L_final = (L_top + L_front) / 2.0
    
    # 3D Math
    volume = Area_top * H_front
    surf_area = 2 * Area_top + (Perim_top * H_front)
    
    # Shape Descriptors
    aspect_ratio = L_top / W_top if W_top > 0 else 0
    elongation = 1.0 / aspect_ratio if aspect_ratio > 0 else 0
    rect_area = L_top * W_top
    compactness = (Perim_top ** 2) / Area_top if Area_top > 0 else 0
    rectangularity = Area_top / rect_area if rect_area > 0 else 0
    eq_diameter = np.sqrt((4 * Area_top) / np.pi)
    
    row = [
        L_final, W_top, H_front,
        Area_top, Perim_top,
        volume, surf_area,
        aspect_ratio, elongation, compactness,
        rectangularity, eq_diameter
    ]
    return np.array(row).reshape(1, -1)

# ==========================================
# Main Flow
# ==========================================
def predict_weight(top_img_path, front_img_path):
    lofo_model, split_model = get_trained_regression_model()
    yolo_model, unet_model, device = load_vision_models()
    
    print("\n[3/3] Analyzing Images...")
    
    # 1. Detect boxes
    top_frame, top_boxes = detect_all_fish(top_img_path, yolo_model)
    front_frame, front_boxes = detect_all_fish(front_img_path, yolo_model)
    
    if len(top_boxes) == 0 or len(front_boxes) == 0:
        print("Error: Could not detect fish in one or both images.")
        return
        
    if len(top_boxes) != len(front_boxes):
        print(f"Warning: YOLO detected {len(top_boxes)} fish in Top view, but {len(front_boxes)} in Front view.")
        print("The script will try to pair them from left to right, but results may be mismatched!")
    
    num_fish = min(len(top_boxes), len(front_boxes))
    print(f"      -> Found {num_fish} paired fish.\n")
    
    # 2. Get full frame masks (run UNet once per image for speed)
    top_raw_mask = get_unet_mask(top_frame, unet_model, device)
    front_raw_mask = get_unet_mask(front_frame, unet_model, device)
    
    # Annotations
    top_annot = top_frame.copy()
    front_annot = front_frame.copy()
    
    for i in range(num_fish):
        # 3. Extract features for this specific fish crop
        t_box = top_boxes[i]
        f_box = front_boxes[i]
        
        t_feat, t_clean = process_fish_crop(top_frame, top_raw_mask, t_box, "top")
        f_feat, f_clean = process_fish_crop(front_frame, front_raw_mask, f_box, "front")
        
        if not t_feat or not f_feat:
            print(f"Skipping Fish #{i+1}: UNet failed to segment the fish inside the bounding box.")
            continue
            
        # Draw on image
        top_annot[t_clean > 0] = [0, 255, 0]
        x1, y1, x2, y2 = [int(v) for v in t_box]
        cv2.rectangle(top_annot, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(top_annot, f"Fish {i+1}", (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        
        front_annot[f_clean > 0] = [0, 255, 0]
        x1, y1, x2, y2 = [int(v) for v in f_box]
        cv2.rectangle(front_annot, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(front_annot, f"Fish {i+1}", (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            
        # 4. Calculate Final Geometry Math
        X_input = calculate_geometry(t_feat, f_feat)
        
        # 5. Predict
        lofo_pred = lofo_model.predict(X_input)[0]
        split_pred = split_model.predict(X_input)[0]
        
        print(f"=============================")
        print(f"FISH #{i+1} PREDICTION:")
        print(f"-----------------------------")
        print(f"Calculated Length : {X_input[0][0]:.2f} cm")
        print(f"Calculated Width  : {X_input[0][1]:.2f} cm")
        print(f"Calculated Height : {X_input[0][2]:.2f} cm")
        print(f"Calculated Volume : {X_input[0][5]:.2f} cm³")
        print(f"-> LOFO REGRESSOR WEIGHT (100% Data)  : {lofo_pred:.2f} g")
        print(f"-> SPLIT REGRESSOR WEIGHT (80% Data)  : {split_pred:.2f} g")
        print(f"=============================\n")

    # Combine images side by side
    top_annot = cv2.addWeighted(top_annot, 0.4, top_frame, 0.6, 0)
    front_annot = cv2.addWeighted(front_annot, 0.4, front_frame, 0.6, 0)
    
    h1, w1 = top_annot.shape[:2]
    h2, w2 = front_annot.shape[:2]
    max_h = max(h1, h2)
    
    if h1 != max_h: top_annot = cv2.resize(top_annot, (w1, max_h))
    if h2 != max_h: front_annot = cv2.resize(front_annot, (w2, max_h))
    
    vis = np.hstack([top_annot, front_annot])
    out_path = os.path.join(ROOT_DIR, "docs", "real_world_prediction_viz.jpg")
    cv2.imwrite(out_path, vis)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict weight from raw images")
    parser.add_argument("--top", required=True, help="Path to Top View image")
    parser.add_argument("--front", required=True, help="Path to Front View image")
    args = parser.parse_args()
    
    predict_weight(args.top, args.front)
