import argparse
import cv2
import time;
from yolo import YOLO
import string
import random
import os
import requests

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.5, help='Confidence for yolo')
args = ap.parse_args()

directory = r'F:\A. Tugas Akhir\A Projek\TA_laravel\public\imgpelanggaran'

classes = ["good", "bad", "none"]

if args.network == "normal":
    print("loading yolov4...")
    yolo = YOLO("models/mask-yolov4.cfg", "models/mask-yolov4.weights", classes)
elif args.network == "prn":
    print("loading yolov3-tiny-prn...")
    yolo = YOLO("models/mask-yolov3-tiny-prn.cfg", "models/mask-yolov3-tiny-prn.weights", classes)
else:
    print("loading yolov4-tiny...")
    yolo = YOLO("models/mask-yolov4-tiny.cfg", "models/mask-yolov4-tiny.weights", classes)

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255)]

print("starting webcam...")
cv2.namedWindow("preview")
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
                
    cv2.imshow("preview", frame)

    rval, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()
