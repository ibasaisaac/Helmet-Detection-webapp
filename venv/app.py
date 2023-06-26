from flask import Flask, render_template, request, Response
import numpy as np
import os
import cv2
import argparse
from PIL import Image
import torch
import datetime
import subprocess
from subprocess import Popen
from re import DEBUG, sub
from werkzeug.utils import send_from_directory


app = Flask(__name__)


@app.route("/")
def welcome():
    return render_template('index.html')


#display the output in the webpage
@app.route('/<path:filename>')
def display_img(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = folder_path+'/'+latest_subfolder
    
    filename = prediction.imgpath
    file_extension = filename.rsplit('.', 1)[1].lower()
    environ = request.environ

    if file_extension == 'jpg' or file_extension == 'png' or file_extension == 'jpeg':      
        return send_from_directory(directory,filename,environ)
    
    else:
        return "Invalid file format"
    

def get_frame():
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    filename = prediction.imgpath    
    image_path = folder_path+'/'+latest_subfolder+'/'+filename    
    video = cv2.VideoCapture(image_path)  # detected video path

    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)   
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        #time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds: 


# play the video in the webpage
@app.route("/video_feed")
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


    
@app.route("/", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        # if "file" not in request.files:
        #     return redirect(request.url)
        if 'file' in request.files:
            file = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',file.filename)
            
            file.save(filepath)
            prediction.imgpath = file.filename

            file_extension = file.filename.rsplit('.', 1)[1].lower()    

            if file_extension == 'jpg' or file_extension == 'png' or file_extension == 'jpeg':
                process = Popen(["python", "detect.py", '--source', filepath, "--weights","best.pt"], shell=True)
                process.wait()
                return display_img(file.filename)
                
                
            elif file_extension == 'mp4':
                process = Popen(["python", "detect.py", '--source', filepath, "--weights","best.pt"], shell=True)
                process.communicate()
                process.wait()
                return video_feed()

            
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    image_path = folder_path+'/'+latest_subfolder+'/'+file.filename 
    return render_template('index.html', image_path=image_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = torch.hub.load('.', 'custom','best.pt', source='local')
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  
