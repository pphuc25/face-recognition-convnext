from flask import Flask, render_template, Response
import convfacenet
import pickle
from scipy.spatial import distance
import numpy as np
import cv2
from PIL import Image

__version__ = "0.1.0"

with open('data/database.pickle', 'rb') as file:
    serialized_data = file.read()
database = pickle.loads(serialized_data)

def predict_pipeline(frame, threshold=0.5):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    PIL_image = Image.fromarray(rgb_small_frame)
    is_face, image_feature = convfacenet.faces_features(PIL_image)
    if is_face == False:
        return "Waiting", 0
    new_face_feature = image_feature[0]

    # similarity_scores = []
    similarity_scores = [(1-distance.cosine(entry['face_feature'], new_face_feature)) for entry in database]
    # for entry in database:
        # similarity = 1 - distance.cosine(entry['face_feature'], new_face_feature)
        # similarity_scores.append(similarity)

    index = np.argmax(similarity_scores)
    predicted_name = database[index]['name']
    confidence = similarity_scores[index]
    if confidence > threshold:
        return predicted_name, confidence 
    return "Unknow", confidence