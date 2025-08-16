import warnings
warnings.filterwarnings('ignore')
from ultralytics.nn.modules.conv import CBAM
from ultralytics import YOLO

if __name__ == '__main__':
    # Initialize model from config for small-object detection
    model = YOLO('ultralytics/cfg/models/11/yolo11s-GoldYOLO.yaml')
    # Uncomment to resume from a checkpoint:
    # model = YOLO('runs/detect/train/weights/best.pt')  # resume from best checkpoint
    # model = YOLO('runs/detect/train/weights/last.pt')  # resume from last checkpoint
    model.load('weights/yolo11s.pt')  # load pretrained weights

    """
    First-round training configuration (kept as reference).
    This block reflects the initial training run and is intentionally preserved.
    All parameter comments are in English and concise.

    results = model.train(
        data='datasets.yaml',        # path to dataset config file (train/val paths, classes)
        task='detect',               # training task: object detection
        resume=False,                # do not resume from previous run (fresh training)
        cache=True,                  # cache images for faster training
        imgsz=640,                   # input image size
        epochs=500,                  # number of training epochs
        batch=4,                     # batch size per GPU
        close_mosaic=0,              # iteration index to stop using mosaic augmentation (0 = disabled)
        workers=16,                  # number of data loader workers
        device='0',                  # CUDA device id or 'cpu'
        optimizer='SGD',             # optimizer type; using fixed SGD instead of 'auto'
        # resume=,                   # set True here to resume training
        amp=False,                   # disable automatic mixed precision (useful if loss becomes NaN)
    )
    """

    # Second-round / current training configuration (active)
    results = model.train(
        data='datasets.yaml',        # path to dataset config (train/val paths and class names)
        task='detect',               # task type: detection
        epochs=500,                  # train sufficiently long
        resume=True,                 # resume from existing checkpoint if available
        batch=8,                     # moderate batch size
        imgsz=704,                   # larger input to improve small-object detection (640 -> 704)
        device='0',                  # CUDA device id (e.g., "0") or "cpu"
        workers=8,                   # dataloader worker threads
        optimizer='SGD',             # use SGD optimizer for stable convergence
        lr0=0.01,                    # initial learning rate
        lrf=0.0001,                  # final learning rate factor
        momentum=0.937,              # SGD momentum
        weight_decay=0.0005,         # weight decay (L2 regularization)
        warmup_epochs=5,             # warmup period in epochs
        cos_lr=True,                 # use cosine annealing learning rate schedule
        seed=42,                     # random seed for reproducibility
        patience=100,                # early stopping patience
        cache=True,                  # cache dataset in memory for speed
        amp=False,                   # disable automatic mixed precision for stability
        mosaic=1.0,                  # enable mosaic augmentation factor
        close_mosaic=350,            # disable mosaic after this epoch (e.g., last 150 epochs disabled)
        mixup=0.0,                   # mixup augmentation factor (disabled)
        copy_paste=0.1,              # copy-paste augmentation probability (small amount helps small objects)
        scale=0.4,                   # scale augmentation range
        translate=0.2,               # translation augmentation range
        hsv_h=0.015,                 # HSV hue augmentation
        hsv_s=0.7,                   # HSV saturation augmentation
        hsv_v=0.4,                   # HSV value augmentation
        fliplr=0.5,                  # horizontal flip probability
        flipud=0.0,                  # vertical flip probability (disabled)
        erasing=0.0,                 # random erasing probability (disabled)
        multi_scale=False,           # disable multi-scale training
        save_period=20,              # checkpoint save period (epochs)
        overlap_mask=True,           # enable overlap mask handling for small objects
        conf=0.001,                  # confidence threshold for logging/detection (low to improve recall)
        iou=0.6,                     # IoU threshold used for NMS during val/test
    )