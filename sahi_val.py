import warnings
warnings.filterwarnings('ignore')
from sahi import AutoDetectionModel
from sahi.predict import predict

if __name__ == '__main__':
    # Load trained model with SAHI
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path='runs/detect/train/weights/best.pt',
        confidence_threshold=0.001,
        device='cuda:0'
    )

    source = 'datasets/test/images/'

    # Run sliced prediction on test set and save results
    predict(
        model_type='ultralytics',
        model_path='runs/detect/train/weights/best.pt',
        model_device='cuda:0',
        model_confidence_threshold=0.001,
        source=source,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        project='sahi_val'
    )