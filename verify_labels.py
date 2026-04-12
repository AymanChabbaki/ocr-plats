import cv2
import os

def visualize_labels(img_path, lbl_path, output_path):
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    
    if not os.path.exists(lbl_path):
        return
        
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            cls_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])
            
            x1 = int((x_c - w/2) * width)
            y1 = int((y_c - h/2) * height)
            x2 = int((x_c + w/2) * width)
            y2 = int((y_c + h/2) * height)
            
            # Draw box
            color = (0, 255, 0) if cls_id > 0 else (255, 0, 0) # Green for chars, Blue for plate
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, str(cls_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(output_path, img)

if __name__ == "__main__":
    os.makedirs("debug", exist_ok=True)
    for i in range(4): # Check our 4 test plates
        for t in ['standard', 'ww', 'w18', 'state']:
            img_name = f"plate_{i}_{t}.jpg"
            lbl_name = img_name.replace(".jpg", ".txt")
            img_path = os.path.join("dataset/images", img_name)
            lbl_path = os.path.join("dataset/labels", lbl_name)
            if os.path.exists(img_path):
                visualize_labels(img_path, lbl_path, os.path.join("debug", f"debug_{img_name}"))
    print("Debug images saved in 'debug/' folder.")
