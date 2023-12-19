# yolo_model.py
from ultralytics import YOLO

def detect_objects_image(image):
    model = YOLO('best.pt')  # load a custom model
    model.predict(image,save=True)
