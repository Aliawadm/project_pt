# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request
import uvicorn
from yolo_model import detect_objects
import os
import re
from fastapi.responses import FileResponse
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
    files = os.listdir(directory)
    prediction_files = [file for file in files if re.search(r'predict\d+', file)]

    if prediction_files:
        predictions = [int(re.search(r'\d+', file).group()) for file in prediction_files]
        latest_prediction = max(predictions)
        latest_prediction_path = os.path.join(directory, f'predict{latest_prediction}')
        return latest_prediction_path
    else:
        return None

@app.post("/")
async def create_upload_file(file: UploadFile = File(...)):
    video_path = 'static/videos/input.mp4'
    with open(video_path, "wb") as video:
        video.write(file.file.read())

    detect_objects(video_path)

    return FileResponse("corrected.mp4", media_type="video/mp4")

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
