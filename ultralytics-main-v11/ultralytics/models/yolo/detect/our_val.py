from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    model = YOLO(r'')
    metrics=model.val(
        val=True,
        data=r'',
        split='test',
        imgsz=640,
        device='',
        workers=4,
        save_json=False,
        save=True,
        save_hybrid=False,
        conf=0.001,
        iou=0.7,
        project='runs/val',
        name='exp',
        plots=True,
    )

    print(f"mAP50-95: {metrics.box.map}") # map50-95
    print(f"mAP50: {metrics.box.map50}")  # map50