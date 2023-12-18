# yolo_model.py
from ultralytics import YOLO
import cv2
import os
import re
import pandas

def detect_objects_image(image):
    model = YOLO('best.pt')  # load a custom model
    model.predict(image,save=True)
