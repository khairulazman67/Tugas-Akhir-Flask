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
# from playsound import playsound
import winsound
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

    classes = ["good", "bad", "1857301038"]
    # datamhs = null
    # response = requests.get('http://127.0.0.1:8000/api/getdatamahasiswa')
    # if response:
    #     print('response',response.text)
    #     datamhs = response.text
    #     print('Input berhasil ',response.data.text)
    # else :
    #     print('gagal')
    # return 
    print("loading yolov3-tiny-prn...")
    yolo = YOLO("metode/models/mask-yolov3-tiny-prn.cfg", "metode/models/mask-yolov3-tiny-prn.weights", classes)

    yolo.size = int(args.size)
    yolo.confidence = float(args.confidence)

    colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255)]

    print("starting webcam...")
    # cv2.namedWindow("preview")
    vc = camera

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

def gen_frames2():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="",
        help="path to (optional) input video file")
    ap.add_argument("-o", "--output", type=str, default="",
        help="path to (optional) output video file")
    ap.add_argument("-d", "--display", type=int, default=1,
        help="whether or not output frame should be displayed")
    args = vars(ap.parse_args())

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
    configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # check if we are going to use GPU
    if config.USE_GPU:
        # set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1]  for i in net.getUnconnectedOutLayers()]

    # initialize the video stream and pointer to output video file
    print("[INFO] accessing video stream...")
    vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
    # vs = camera
    writer = None

    # loop over the frames from the video stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # resize the frame and then detect people (and only people) in it
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln,
            personIdx=LABELS.index("person"))

        # initialize the set of indexes that violate the minimum social
        # distance
        violate = set()

        # ensure there are *at least* two people detections (required in
        # order to compute our pairwise distance maps)
        if len(results) >= 2:
            # extract all centroids from the results and compute the
            # Euclidean distances between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two
                    # centroid pairs is less than the configured number
                    # of pixels
                    if D[i, j] < config.MIN_DISTANCE:
                        # update our violation set with the indexes of
                        # the centroid pairs
                        violate.add(i)
                        violate.add(j)

        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract the bounding box and centroid coordinates, then
            # initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # if the index pair exists within the violation set, then
            # update the color
            if i in violate:
                print('terdeteksi')
                # playsound('alarm.mp3')
                freq = 100
                dur = 50
                for i in range(0, 5):    
                    winsound.Beep(freq, dur)    
                    freq+= 100
                    dur+= 50
                color = (0, 0, 255)

            # draw (1) a bounding box around the person and (2) the
            # centroid coordinates of the person,
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

        # draw the total number of social distancing violations on the
        # output frame
        text = "Pelanggar Physical Distancing Terdeteksi: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

        # check to see if the output frame should be displayed to our
        # screen
        if args["display"] > 0:
            # show the output frame
            # cv2.imshow("Frame", frame)
            
            grabbed,buffer =cv2.imencode('.jpg', frame)
            grabbed, frame = vs.read()
            frame2=buffer.tobytes()
            yield(b'--frame\r\n'
				b'Content-Type:image/jpeg\r\n\r\n'+ frame2 + b'\r\n')	
    
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # if an output video file path has been supplied and the video
        # writer has not been initialized, do so now
        if args["output"] != "" and writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 25,
                (frame.shape[1], frame.shape[0]), True)

        # if the video writer is not None, write the frame to the output
        # video file
        if writer is not None:
            writer.write(frame)

@app.route("/")
def index():
	return render_template("monitor.html")

@app.route("/monitor")
def monitor():
	return render_template("monitor2.html")

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
    return Response(gen_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
	app.run(debug=True)