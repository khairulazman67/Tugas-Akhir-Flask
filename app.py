from flask import Flask, render_template, Response
# face mask
from metode.yolo import YOLO
# Social Distancing
from metode.pytool import social_distancing_config as config
from metode.pytool.detection import detect_people
from scipy.spatial import distance as dist

import numpy as np
import imutils

import cv2

import argparse
import time;

import string
import random
import os
import requests
# from flask_assets import Bundle, Environment

app = Flask(__name__)


camera=cv2.VideoCapture(0)

def gen_frames1():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn')
    ap.add_argument('-d', '--device', default=0, help='Device to use')
    ap.add_argument('-s', '--size', default=416, help='Size for yolo')
    ap.add_argument('-c', '--confidence', default=0.5, help='Confidence for yolo')
    args = ap.parse_args()

    directory = r'F:\A. Tugas Akhir\A Projek\TA_laravel\public\imgpelanggaran'

    classes = ["good", "bad", "none"]

    print("loading yolov3-tiny-prn...")
    yolo = YOLO("metode/models/mask-yolov3-tiny-prn.cfg", "metode/models/mask-yolov3-tiny-prn.weights", classes)

    yolo.size = int(args.size)
    yolo.confidence = float(args.confidence)

    colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255)]

    print("starting webcam...")
    # cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    os.chdir(directory)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        width, height, inference_time, results = yolo.inference(frame)
        for detection in results:
            id, name, confidence, x, y, w, h = detection
            cx = x + (w / 2)
            cy = y + (h / 2)
            
            # draw a bounding box rectangle and label on the image
            color = colors[id]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
            # print(frame)
            if name!="good":
                # print(time.localtime(time.time()))    
                N = 7
                res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k = N))
                nama_file = str(res)+'.jpg'
                
                data = {
                    'NIM' : '1857301038',
                    'bukti' : nama_file
                }
                print(data)
                response = requests.post('http://127.0.0.1:8000/api/inputpelanggaran', data = data)
                if response:
                    print('Input berhasil ',response.text)
                    cv2.imwrite(nama_file,frame)    
                else :
                    print('input gagal ',response.text)
        
    
        # cv2.imshow("preview", frame)
        rval,buffer =cv2.imencode('.jpg', frame)
        rval, frame = vc.read()
        frame2=buffer.tobytes()

        yield(b'--frame\r\n'
				b'Content-Type:image/jpeg\r\n\r\n'+ frame2 + b'\r\n')	
    
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break


    cv2.destroyWindow("preview")
    vc.release()
    
    # while True:
	# 	## read the camera frame
    #     success,frame =camera.read()
    #     if not success:
	# 		break
    #     else:
	# 		ret,buffer =cv2.imencode('.jpg', frame)
	# 		frame=buffer.tobytes()

	# 	yield(b'--frame\r\n'
	# 			b'Content-Type:image/jpeg\r\n\r\n'+ frame + b'\r\n')	

@app.route("/")
def index():
	return render_template("monitor.html")

@app.route("/berandaStaf")
def berandaStaf():
	return render_template("berandaStaf.html")

@app.route("/dataMhs")
def dataMhs():
	return render_template("dataMhs.html")

@app.route("/login")
def login():
	return render_template("login.html")

@app.route("/detailPelanggaran")
def detailPelanggaran():
	return render_template("detailPelanggaran.html")



@app.route('/video_feed1')
def video_feed1():
    return Response(gen_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
	app.run(debug=True)