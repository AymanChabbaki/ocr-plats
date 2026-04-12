from ultralytics import YOLO
import torch

def train_model():
    # Load a model
    # 'yolo11s.pt' is a good balance for license plates. 
    # Use 'yolo11n.pt' for speed or 'yolo11m.pt' for better accuracy.
    model = YOLO("yolo11s.pt")

    # Hardware Check
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Train the model
    # H100 optimizations: 
    # - batch=64 (or -1 for auto-batching to fit memory)
    # - amp=True (Automatic Mixed Precision)
    # - imgsz=1024 (Leveraging our high-res upscaled plates)
    results = model.train(
        data="data.yaml", 
        epochs=100,
        imgsz=1024,
        batch=64,           # Adjust this if you get Out Of Memory (OOM)
        device=device,
        amp=True,           # Faster training on H100
        workers=8,          # Number of CPU cores for data loading
        project="moroccan_plates",
        name="h100_training",
        exist_ok=True,
        augment=True,       # Enable additional online augmentations
        patience=20         # Early stopping if no improvement for 20 epochs
    )

if __name__ == "__main__":
    train_model()
