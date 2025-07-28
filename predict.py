from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')  # model path

# Set test image or folder path
source = 'datasets/test/images/'  # image or folder path

# Run prediction and save results
model.predict(source, save=True)