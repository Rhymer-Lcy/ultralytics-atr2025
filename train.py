import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11-p2.yaml')  # model config for small objects
    model.load('weights/yolo11m.pt')  # load pretrained weights

    results = model.train(
        data='datasets.yaml',    # dataset yaml path
        epochs=200,              # training epochs
        batch=16,                # batch size
        imgsz=(640, 512),               # image size, match input width
        workers=8,               # dataloader workers
        device=0,                # GPU device id
        optimizer='SGD',         # optimizer type
        amp=True,                # enable mixed precision
        cache=True,              # cache images in RAM
        conf=0.001,              # low conf threshold for small objects
        iou=0.5,                 # IoU threshold for mAP
        hsv_h=0.015,             # color augmentation
        hsv_s=0.75,              # color augmentation
        hsv_v=0.05,              # color augmentation
        degrees=0.0,             # rotation augmentation
        translate=0.1,           # translation augmentation
        scale=0.05,              # scale augmentation
        shear=0.0,               # shear augmentation
        flipud=0.5,              # vertical flip
        fliplr=0.5,              # horizontal flip
        mosaic=0.5,              # mosaic augmentation
        close_mosaic=10          # disable mosaic in last 10 epochs
    )