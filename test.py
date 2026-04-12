from ultralytics import YOLO
import os
import glob

def run_test():
    # Load the best trained model
    # Usually found in runs/detect/h100_training/weights/best.pt
    model_path = "moroccan_plates/h100_training/weights/best.pt"
    
    if not os.path.exists(model_path):
        # Fallback to the pretrained model for demonstration if training hasn't finished
        print(f"Warning: {model_path} not found. Using pretrained yolo11s.pt for demo.")
        model = YOLO("yolo11s.pt")
    else:
        model = YOLO(model_path)

    # Path to test images
    test_images = glob.glob("dataset/images/val/*.jpg")[:10] # Test on first 10 images
    
    if not test_images:
        print("No validation images found in dataset/images/val/.")
        return

    # Run inference
    # imgsz should match training imgsz
    results = model.predict(
        source=test_images,
        imgsz=1024,
        save=True,
        project="test_results",
        name="inference_run",
        exist_ok=True
    )

    print(f"Inference complete! Check the 'test_results/inference_run' folder for visualized detections.")

if __name__ == "__main__":
    run_test()
