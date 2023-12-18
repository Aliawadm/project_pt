# yolo_model.py
from ultralytics import YOLO
import cv2
import os
import re
from moviepy.editor import VideoFileClip
import re
from moviepy.editor import *
from collections import defaultdict
import numpy
def find_latest_prediction(directory):
    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        return None

    
    prediction_files = [file for file in files if re.search(r'predict\d+', file)]

    if prediction_files:
        
        predictions = [int(re.search(r'\d+', file).group()) for file in prediction_files]
        latest_prediction = max(predictions)

        
        latest_prediction_path = os.path.join(directory, f'predict{latest_prediction}')

        return latest_prediction_path
    else:
        return None 

def detect_objects(video_path):
    cap = cv2.VideoCapture(video_path)

    model = YOLO('best.pt')  
    
    output_video_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    f_num = 0
    d={1:0}
    q=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        results=model.track(frame,conf=0.19,persist=True, save=True)

        
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            z=boxes.xyxy.tolist()
            zz=boxes.xyxyn.tolist()
        if boxes.id is None:
            pass
        else:
            q = boxes.id
            w=int(q[0])
            q=w
            if q not in d.keys():
                d[q]= 0


        
        
        maxPrice=8751 
        image_path=find_latest_prediction(r"C:\Users\HP\Desktop\project_pt\runs\detect")
        complete_file_path = os.path.join(image_path, 'image0.jpg')
        image = cv2.imread(complete_file_path)
        for i in range(len(z)):
            height = zz[i][3] - zz[i][1]
            width = zz[i][2] - zz[i][0]
            costt=  round((height*width)*maxPrice, 4)                       #(height*width)(((zz[i][2] - zz[i][0])+(zz[i][3] - zz[i][1])) * maxPrice)
            #pot[idP[0]]=costt
            #text_position = ((z[i][0] + z[i][2]) // 2, z[i][1] - 10)
            # Extract coordinates
            x_coordinate = int(((z[i][0] + z[i][2]) // 2)-40)
            y_coordinate = int((z[i][1] - 10)-20)
            text_position = (x_coordinate, y_coordinate)
            if  not isinstance(text_position, tuple) or len(text_position) != 2:
                raise ValueError("Invalid text position format")
            xmin = x_coordinate 
            xmax = x_coordinate+200
            ymin = y_coordinate -30
            ymax = y_coordinate +10
            top_left_corner = (int(xmin),int(ymin))
            bottom_right_corner = (int(xmax), int(ymax))
            cv2.rectangle(image, top_left_corner, bottom_right_corner, (255, 255, 255, 255), cv2.FILLED)
            cv2.putText(image, "{} SAR".format(costt), text_position, cv2.FONT_ITALIC, 0.9, (0, 0, 0), 2)
            #print(i,', The cost is: ',costt)
        if q>0:
          if d[q]<costt:
              d[q]=costt

        cv2.imwrite(complete_file_path, image)
        video_writer.write(image)
        f_num += 1
    
  
    # Load images
    base_image = image
    overlay_image = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)  # Use IMREAD_UNCHANGED to keep alpha channel if present

    # Resize overlay image if needed
    overlay_image = cv2.resize(overlay_image, (200, 200))

    # Define ROI (Region of Interest) in the base image
    rows, cols, channels = overlay_image.shape
    roi = base_image[0:rows, 0:cols]

    # Create a mask and inverse mask of the overlay image
    mask = overlay_image[:, :, 3]  # Assuming the fourth channel is the alpha channel
    mask_inv = cv2.bitwise_not(mask)

    # Extract RGB channels from overlay image
    overlay_rgb = overlay_image[:, :, 0:3]

    # Extract ROI from base image
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Extract region of overlay image
    roi_fg = cv2.bitwise_and(overlay_rgb, overlay_rgb, mask=mask)

    # Combine the two images
    


    
    cv2.rectangle(base_image, (0,0), (1400,1200), (255, 255, 255, 255), cv2.FILLED)
    g=250
    for i in d.keys():
        cv2.putText(base_image, "Cost: {} SAR".format(d[i]), (320,g), cv2.FONT_ITALIC, 0.9, (0, 0, 0), 2)
        g=g+50
        
    
    dst = cv2.add(roi_bg, roi_fg)
    base_image[0:rows, 0:cols] = dst
    cv2.imwrite(complete_file_path, base_image)
    video_writer.write(base_image)
    cap.release()
    video_writer.release()  
    vid = VideoFileClip("output_video.mp4")
    vid.write_videofile("corrected.mp4")





detect_objects(R"C:\Users\HP\Desktop\project_pt\static\videos\input.mp4")