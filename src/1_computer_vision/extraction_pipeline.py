import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import re

# ==========================================
# Configuration
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRAMES_DIR = os.path.join(ROOT_DIR, "data", "images") # Add your input images here
YOLO_MODEL_PATH = os.path.join(ROOT_DIR, "models", "weights", "yolo_v12m_best.pt")
UNET_MODEL_PATH = os.path.join(ROOT_DIR, "models", "weights", "unet_mit_b0_best.pt")
OUTPUT_CSV = os.path.join(ROOT_DIR, "data", "fish_frames_unet.csv")
MASKS_DIR = os.path.join(ROOT_DIR, "data", "production_masks_output")

# Read true weights and dimensions dynamically
def load_ground_truth():
    weights_path = os.path.join(ROOT_DIR, "weights.csv")
    truth_path = os.path.join(ROOT_DIR, "fish_truth.csv")
    
    true_weights = {}
    if os.path.exists(weights_path):
        df_w = pd.read_csv(weights_path)
        for _, row in df_w.iterrows():
            fid = str(row["FishID"]).strip().lower()
            # Normalize fish1, fish2 to fish01, fish02
            match = re.search(r'fish(\d+)', fid)
            if match:
                num = int(match.group(1))
                fid = f"fish{num:02d}"
            true_weights[fid] = float(row["Weight"])
            
    true_lengths, true_heights = {}, {}
    if os.path.exists(truth_path):
        df_t = pd.read_csv(truth_path)
        for _, row in df_t.iterrows():
            fid = str(row["Fish"]).strip().lower()
            # Normalize fish1, fish2 to fish01, fish02
            match = re.search(r'fish(\d+)', fid)
            if match:
                num = int(match.group(1))
                fid = f"fish{num:02d}"
            true_lengths[fid] = float(row["Length"])
            true_heights[fid] = float(row["Height"])
            
    return true_weights, true_lengths, true_heights

TRUE_WEIGHTS, TRUE_LENGTHS, TRUE_HEIGHTS = load_ground_truth()

# Tape Calibration Constants (px to cm)
TOP_PX_PER_CM = 60.36 / 10.0
FRONT_PX_PER_CM = 65.35 / 10.0

def get_fish_id_and_view(folder_name):
    folder_name = folder_name.lower()
    view = "top" if "top" in folder_name else "front"
    nums = re.findall(r'\d+', folder_name)
    if nums:
        num = int(nums[0])
        fish_id = f"fish0{num}" if num < 10 and num == 1 else f"fish{num}"
        return fish_id, view
    return None, None

def remove_fins(mask, bbox):
    """Dynamic elliptical morphological opening to remove fins/tails without eroding body."""
    x1, y1, x2, y2 = bbox
    k_size = max(3, int(min(x2-x1, y2-y1) * 0.15))
    if k_size % 2 == 0: k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Keep only the largest component to drop disconnected fins
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        final = np.zeros_like(clean)
        cv2.drawContours(final, [c], -1, 255, thickness=cv2.FILLED)
        return final
    return clean

def extract_features(mask, view="top"):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    
    if view == "top":
        rect = cv2.minAreaRect(c)
        (x, y), (w, h), angle = rect
        length = max(w, h)
        width = min(w, h)
    else:
        # Front view: Strict Vertical/Horizontal Pixel Span
        x, y, w, h = cv2.boundingRect(c)
        length = max(w, h)
        width = min(w, h)
        
    return {"length_px": length, "width_px": width, "area_px": area, "perimeter_px": perimeter, "mask_px": np.sum(mask > 0)}

def detect_fish(img_path, yolo_model):
    """Run YOLO detection and return frame, best box, and confidence."""
    frame = cv2.imread(img_path)
    if frame is None: return None, None, None
    res = yolo_model(frame, verbose=False, imgsz=640)
    boxes = res[0].boxes.xyxy.cpu().numpy()
    confs = res[0].boxes.conf.cpu().numpy()
    if len(boxes) == 0: return frame, None, None
    best_idx = np.argmax(confs)
    return frame, boxes[best_idx], confs[best_idx]

def get_unet_mask(frame, unet_model, device):
    """Run UNet segmentation on the full frame and return a mask."""
    h, w = frame.shape[:2]
    
    # Albumentations transform for validation
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

def clip_mask_to_box(mask, box):
    """STRICTLY clip mask to only keep pixels INSIDE the YOLO bounding box."""
    x1, y1, x2, y2 = [int(v) for v in box]
    clipped = np.zeros_like(mask)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped

def process_frame(img_path, yolo_model, unet_model, device, view="top"):
    """Process a single frame with UNet."""
    frame, box, conf = detect_fish(img_path, yolo_model)
    if frame is None or box is None: return None, None
    
    raw_mask = get_unet_mask(frame, unet_model, device)
    
    # STRICT: Only keep mask pixels inside the YOLO bounding box
    raw_mask = clip_mask_to_box(raw_mask, box)
    
    clean_mask = remove_fins(raw_mask, box)
    
    features = extract_features(clean_mask, view)
    if features:
        features["Score"] = conf
        x1, y1, x2, y2 = [int(v) for v in box]
        annotated = frame.copy()
        annotated[clean_mask > 0] = [0, 255, 0]
        annotated = cv2.addWeighted(annotated, 0.4, frame, 0.6, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return features, annotated
    return None, None

def run_pipeline():
    print("Finding relaxed synced pairs (MAX_FRAME_DIFF=50)...")
    fish_frames = {}
    
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for path in glob.glob(os.path.join(FRAMES_DIR, "**", ext), recursive=True):
            if "runs" in path:
                continue
                
            filename = os.path.basename(path).lower()
            
            fish_match = re.search(r'fish_?(\d+)', filename)
            frame_match = re.search(r'frame_(\d+)', filename)
            
            if fish_match and frame_match:
                num = int(fish_match.group(1))
                fish_id = f"fish{num:02d}"
                frame_num = int(frame_match.group(1))
                
                view = None
                if "top" in filename: view = "top"
                elif "front" in filename: view = "front"
                
                if view:
                    if fish_id not in fish_frames: 
                        fish_frames[fish_id] = {"top": [], "front": []}
                    fish_frames[fish_id][view].append((frame_num, path))

    paired_paths = []
    MAX_FRAME_DIFF = 0
    for fish_id, views in fish_frames.items():
        top_list = sorted(views["top"], key=lambda x: x[0])
        front_list = sorted(views["front"], key=lambda x: x[0])
        
        paired_front_indices = set()
        for t_frame, t_path in top_list:
            best_diff = float('inf')
            best_f_idx = -1
            
            for i, (f_frame, f_path) in enumerate(front_list):
                if i in paired_front_indices:
                    continue
                diff = abs(t_frame - f_frame)
                if diff < best_diff:
                    best_diff = diff
                    best_f_idx = i
                    
            if best_f_idx != -1 and best_diff <= MAX_FRAME_DIFF:
                paired_front_indices.add(best_f_idx)
                f_frame, f_path = front_list[best_f_idx]
                paired_paths.append((fish_id, t_frame, t_path, f_path))
                
    print(f"Found {len(paired_paths)} perfect Top/Front synced pairs!")
    if not paired_paths:
        return

    print("Loading Models...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        unet_model = smp.Unet(encoder_name="mit_b0", encoder_weights=None, in_channels=3, classes=1)
    except:
        unet_model = smp.Unet(encoder_name="efficientnet-b3", encoder_weights=None, in_channels=3, classes=1)
    unet_model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=device))
    unet_model.to(device)
    unet_model.eval()

    os.makedirs(MASKS_DIR, exist_ok=True)
    raw_paired_data = []

    for fish_id, frame_num, top_path, front_path in tqdm(paired_paths, desc="Processing Paired Frames"):
        top_mask_path = os.path.join(MASKS_DIR, f"{fish_id}_frame_{frame_num}_top.jpg")
        front_mask_path = os.path.join(MASKS_DIR, f"{fish_id}_frame_{frame_num}_front.jpg")
        if os.path.exists(top_mask_path) and os.path.exists(front_mask_path):
            top_features, top_img = process_frame(top_path, yolo_model, unet_model, device, view="top")
            front_features, front_img = process_frame(front_path, yolo_model, unet_model, device, view="front")
            if top_features and front_features:
                raw_paired_data.append({
                    "FishID": fish_id, "FrameIndex": frame_num,
                    "Top_Image": top_path, "Front_Image": front_path,
                    "Top_Length_px": top_features["length_px"], "Top_Width_px": top_features["width_px"],
                    "Top_Area_px": top_features["area_px"], "Top_Perim_px": top_features["perimeter_px"],
                    "Top_MaskPx": top_features["mask_px"], "Top_Score": top_features["Score"],
                    "Front_Length_px": front_features["length_px"],
                    "Front_Height_px": front_features["width_px"],
                    "Front_MaskPx": front_features["mask_px"], "Front_Score": front_features["Score"]
                })
            continue
            
        top_features, top_img = process_frame(top_path, yolo_model, unet_model, device, view="top")
        front_features, front_img = process_frame(front_path, yolo_model, unet_model, device, view="front")
        
        if top_features and front_features:
            raw_paired_data.append({
                "FishID": fish_id,
                "FrameIndex": frame_num,
                "Top_Image": top_path,
                "Front_Image": front_path,
                "Top_Length_px": top_features["length_px"],
                "Top_Width_px": top_features["width_px"],
                "Top_Area_px": top_features["area_px"],
                "Top_Perim_px": top_features["perimeter_px"],
                "Top_MaskPx": top_features["mask_px"],
                "Top_Score": top_features["Score"],
                "Front_Length_px": front_features["length_px"],
                "Front_Height_px": front_features["width_px"],
                "Front_MaskPx": front_features["mask_px"],
                "Front_Score": front_features["Score"]
            })
            
            cv2.imwrite(top_mask_path, top_img)
            cv2.imwrite(front_mask_path, front_img)

    raw_df = pd.DataFrame(raw_paired_data)
    
    # --- CALIBRATION ---
    print("\nCalibrating Pixels to Centimeters using tape measurements...")
    final_rows = []
    
    for fish_id, group in raw_df.groupby("FishID"):
        true_weight = TRUE_WEIGHTS.get(fish_id, np.nan)
        true_length = TRUE_LENGTHS.get(fish_id, np.nan)
        true_height = TRUE_HEIGHTS.get(fish_id, np.nan)
        
        for _, row in group.iterrows():
            length_cm = row["Top_Length_px"] / TOP_PX_PER_CM
            width_cm = row["Top_Width_px"] / TOP_PX_PER_CM
            height_cm = row["Front_Height_px"] / FRONT_PX_PER_CM
            area_cm2 = row["Top_Area_px"] / (TOP_PX_PER_CM ** 2)
            perim_cm = row["Top_Perim_px"] / TOP_PX_PER_CM
            
            # Data Quality Filters
            if length_cm <= 0 or width_cm <= 0 or height_cm <= 0:
                continue
                
            if height_cm > length_cm:
                continue
                
            volume = np.pi * length_cm * (width_cm / 2.0) * (height_cm / 2.0)
            surface_area = 2 * area_cm2 + perim_cm * height_cm
            aspect_ratio = length_cm / width_cm if width_cm > 0 else 0
            rect = area_cm2 / (length_cm * width_cm) if (length_cm * width_cm) > 0 else 0
            eq_diam = np.sqrt(4 * area_cm2 / np.pi)
            
            final_rows.append({
                "FishID": fish_id,
                "Weight (g)": true_weight,
                "True_Length_cm": true_length,
                "True_Height_cm": true_height,
                "FrameIndex": row["FrameIndex"],
                "Top_Image_File": row["Top_Image"],
                "Front_Image_File": row["Front_Image"],
                "Timestamp (s)": 0, 
                "FPS_Top": 20,
                "FPS_Front": 20,
                "Length (cm)": length_cm,
                "Width (cm)": width_cm,
                "Height (cm)": height_cm,
                "Area (cm²)": area_cm2,
                "Perimeter (cm)": perim_cm,
                "TopMaskPixels": row["Top_MaskPx"],
                "FrontMaskPixels": row["Front_MaskPx"],
                "BlurTop": 500,
                "BlurFront": 500,
                "Score": (row["Top_Score"] + row["Front_Score"]) / 2,
                "Volume (cm³)": volume,
                "Surface Area (cm²)": surface_area,
                "Aspect Ratio": aspect_ratio,
                "Elongation": 1.0 / aspect_ratio if aspect_ratio > 0 else 0,
                "Compactness": (perim_cm ** 2) / area_cm2 if area_cm2 > 0 else 0,
                "Condition Factor (K)": (true_weight / (length_cm ** 3)) * 100 if length_cm > 0 else 0,
                "Rectangularity": rect,
                "Equivalent Diameter (cm)": eq_diam
            })
            
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nProduction pipeline finished! Dataset saved to: {OUTPUT_CSV}")
    print(f"Total synced rows: {len(final_df)}")

if __name__ == '__main__':
    run_pipeline()

