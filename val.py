import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # Load trained model
    model = YOLO('runs/detect/train/weights/best.pt')  # your trained model path

    # Evaluate on test set (use split='val' for validation set)
    model.val(
        data='datasets.yaml', # your dataset yaml file
        split='test',         # 'test' or 'val'
        imgsz=512,            # match training image size
        batch=16,             # match training batch size
        workers=8,            # match training workers
        iou=0.6,              # IoU threshold for mAP
        conf=0.001,           # confidence threshold
    )