import sys
import convfacenet
import pickle
from scipy.spatial import distance
import numpy as np
import cv2
from PIL import Image
import os

from manipulate_data import InputData


# current_dir = os.getcwd()
# dir_code = 'src'
# data_dir = 'data/database.pickle'

# code_path = os.path.join(current_dir, dir_code)
# sys.path.append(code_path)  # Add the directory to the module search path
# data_path = os.path.join(current_dir, data_dir)


__version__ = "0.1.0"


config = InputData.load_config()
database = InputData.load_database(config)


def predict_pipeline(frame, threshold=0.67):
    if frame is None:
        return "Unknow", 0
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    PIL_image = Image.fromarray(rgb_small_frame)
    is_face, image_features = convfacenet.faces_features(PIL_image)
    if is_face == False:
        return [("Waiting", 0)]

    predicted_results = []
    for image_feature in image_features:
        new_face_feature = image_feature

        similarity_scores = [(1-distance.cosine(entry['face_feature'], new_face_feature)) for entry in database]

        index = np.argmax(similarity_scores)
        predicted_name = database[index]['name']
        confidence = similarity_scores[index]
        if confidence > threshold:
            predicted_results.append((predicted_name, confidence))
        else:
            predicted_results.append(("Unknow", confidence))
    return predicted_results