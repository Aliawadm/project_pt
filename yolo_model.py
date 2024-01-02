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
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle,Image,Spacer
import shutil
import matplotlib.pyplot as plt

def graph(data):
    m = data[1:]
    total = 0
    for i in range(len(m)):
        total+=m[i][1]
    plt.bar(
        x=[2024,2025,2026,2027,2028,2029],
        height=[total,total+(total*0.25),total+(total*0.5),total+(total*0.75),total*2,total*3],
        color="#0a731d"
    )
    plt.title('The total cost in 5 years')
    plt.savefig('pltTest.png', bbox_inches='tight')
    

def FullReport(file_path,data):
    graph(data)
    m = data[1:]
    total = 0
    for i in range(len(m)):
        total+=m[i][1]
    data.append(["Total :",total,"-","-"])
    pdf = SimpleDocTemplate(file_path, pagesize=letter, leftMargin=20, rightMargin=20, topMargin=20, bottomMargin=20)

    col_widths = [pdf.width / len(data[0])] * len(data[0])

    table = Table(data, colWidths=col_widths)

    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), '#8ec73f'),
        ('TEXTCOLOR', (0, 0), (-1, 0), (1, 1, 1)), # Header text color
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), '#000000'),
        ('GRID', (0, 0), (-1, -1), 1, '#000000'),
        ('TEXTCOLOR', (0, 1), (-1, -1), (1, 1, 1)),  # Data text color
    ])
    
    table.setStyle(style)
    image_path = 'logo.png'
    header_image = Image(image_path, width=100, height=70)

    spacer=Spacer(1,20)

    graph_im = 'pltTest.png'
    g_image = Image(graph_im, width=300, height=300)

    pdf.build([header_image,table,spacer,g_image])
    
    static_file_path = os.path.join(os.getcwd(),"static")
    destination_path = os.path.join(os.getcwd(),"static\Fatorh.pdf")
    if os.path.exists(destination_path):
      os.remove(destination_path)  # Remove the existing file
    shutil.move(file_path, static_file_path)


def find_latest_prediction(directory):
    try:
        complete_file_path = os.path.join(os.getcwd(),directory)
        files = os.listdir(complete_file_path)
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
    D=[['potholes ID','Cost','height','width']]
    d={1:[0,0,0]}
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
                d[q]= [0,0,0]


        
        
        maxPrice=8751 
        directory_path = r"runs\detect"
        image_path=find_latest_prediction(directory_path)
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
          if d[q][0]<costt:
              d[q][0]=costt
              d[q][1]=height
              d[q][2]=width

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
        cv2.putText(base_image, "Cost: {} SAR".format(d[i][0]), (320,g), cv2.FONT_ITALIC, 0.9, (0, 0, 0), 2)
        g=g+50
        
    
    dst = cv2.add(roi_bg, roi_fg)
    base_image[0:rows, 0:cols] = dst
    cv2.imwrite(complete_file_path, base_image)
    video_writer.write(base_image)
    cap.release()
    video_writer.release()  
    Fatorh_path = os.path.join(os.getcwd(),"Fatorh.pdf")
    for i in d.keys():
      D.append([i,d[i][0],d[i][1],d[i][2]])
    FullReport(Fatorh_path,D)
    vid = VideoFileClip("output_video.mp4")
    vid.write_videofile("corrected.mp4")

