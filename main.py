# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request
import uvicorn
from yolo_model import detect_objects
import os
import re
from yolo_model_image import detect_objects_image
from fastapi.responses import FileResponse
from PIL import Image
from fastapi.responses import StreamingResponse
from io import BytesIO
from fastapi.staticfiles import StaticFiles
app = FastAPI(debug=True)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class FileUpload(BaseModel):
    file: UploadFile

@app.get("/")
def home(request: Request):
    video_url = "/static/back.mp4"
    return templates.TemplateResponse("index.html", {"request": request,"video_url": video_url})

def find_latest_prediction(directory):
    complete_file_path = os.path.join(os.getcwd(),directory)
    files = os.listdir(complete_file_path)
    prediction_files = [file for file in files if re.search(r'predict\d+', file)]

    if prediction_files:
        predictions = [int(re.search(r'\d+', file).group()) for file in prediction_files]
        latest_prediction = max(predictions)
        latest_prediction_path = os.path.join(directory, f'predict{latest_prediction}')
        return latest_prediction_path
    else:
        return None

@app.post("/")
def create_upload_file(file: UploadFile = File(...)):
    video_path = 'static/videos/input.mp4'
    with open(video_path, "wb") as video:
        video.write(file.file.read())

    detect_objects(video_path)

    return FileResponse("corrected.mp4", media_type="video/mp4")

@app.get("/image")
def home(request: Request):
    video_url = "/static/back.mp4"
    return templates.TemplateResponse("index_image.html", {"request": request,"video_url": video_url})

@app.post("/image")
def create_upload_file(file: UploadFile = File(...)):
    image_path = 'static/images/input.jpg'
    with open(image_path, "wb") as image:
        image.write(file.file.read())

    detect_objects_image(image_path)
    directory_path = r"runs\detect"
    latest_prediction_path = find_latest_prediction(directory_path)
    complete_file_path = os.path.join(latest_prediction_path, 'input.jpg')
    # Open image
    image = Image.open(complete_file_path)

    # Convert image to BytesIO object
    img_io = BytesIO()
    image.save(img_io, format='JPEG')
    img_io.seek(0)

    # Return the image data in the response
    return StreamingResponse(img_io, media_type="image/jpeg")

@app.get("/analysis")
def home(request: Request):
    video_url = "/static/back.mp4"
    pdf_url = "/static/Fatorh.pdf"
    return templates.TemplateResponse("index_analysis.html", {"request": request,"video_url": video_url,"pdf_url":pdf_url})

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
