import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11m.yaml')  # model config path
    model.load('weights/yolo11m.pt')  # load pretrained weights if needed
    results = model.train(
        data='datasets.yaml',    # dataset yaml path
        epochs=200,              # training epochs
        batch=16,                # batch size
        imgsz=512,               # image size
        workers=8,               # dataloader workers
        device=0,                # GPU device id
        optimizer='SGD',         # optimizer type
        amp=True,                # enable mixed precision
        cache=True               # cache images in RAM
    )