from sahi import AutoDetectionModel
from sahi.predict import predict

# Load trained model with SAHI
detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path='runs/detect/train/weights/best.pt',
    confidence_threshold=0.3,
    device='cuda:0'
)

source = 'datasets/test/images/'

# Run sliced prediction and save results
predict(
    model_type='ultralytics',
    model_path='runs/detect/train/weights/best.pt',
    model_device='cuda:0',
    model_confidence_threshold=0.3,
    source=source,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    project='SAHI_predict'
)