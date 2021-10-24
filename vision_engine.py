import tensorflow_hub as hub
import cv2
import numpy
import tensorflow as tf
import pandas as pd
from flask import Flask, json, request, jsonify, send_file
import os
import matplotlib.pyplot as plt

width = 512
height = 512

## Loading pre-trained object detection models and labels
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
labels = pd.read_csv('testingdata/labels.csv',sep=';',index_col='ID')
labels = labels['OBJECT (2017 REL.)']

## Preparing the Flask API
app = Flask(__name__)
app.config['UPLOAD_FOLDER']=os.getcwd() + "/uploads"
ALLOWED_EXTENSIONS = {"jpg", "png", "jpeg"}

## Function to allowlist only jpeg or png files
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(frame):
    id=0
    return_dict={}
    frame_resized=cv2.resize(frame,(width,height))
    rgb=cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    boxes, scores, classes, num_detections = detector(rgb_tensor)

    # Processing outputs
    pred_labels = classes.numpy().astype('int')[0] 
    pred_labels = [labels[i] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]

    for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.5:
            continue
        score_txt = f'{100 * round(score)}%'
        return_dict[id]={"confidence":score_txt,"box":list(map(int,[ymin,xmin,ymax,xmax])),"label":label}
        id=id+1
    return jsonify(return_dict)




@app.route("/detect",methods=["POST"])
def get_frame():
    #if "file" not in request.files:
        #return(jsonify({"error":"Image file NOt Found"}))
    file = request.files["upload_file"]
    img = cv2.imdecode(numpy.fromstring(request.files['upload_file'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
    return process_image(img)


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)