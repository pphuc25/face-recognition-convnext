import os

import cv2
import gdown

import sys
sys.path.append('/Users/user/PycharmProjects/FPT code/2023/Summer2023/DPL302m/Multi-Face-Recognize/src')
from convfacenet import utils
import numpy as np
from PIL import Image
import math
from convfacenet.face_detector.detector import RetinaFace
import torch


def alignment_procedure(img, left_eye, right_eye):
    """
    this function aligns given face in img based on left and right eye coordinates
    :param img: face-image
    :param left_eye: tuple(x,y) left_eye coordinates
    :param right_eye: tuple(x,y) right_eye coordinates
    :return: aligned image
    """

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # -----------------------
    # find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = utils.euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = utils.euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = utils.euclidean_distance(np.array(right_eye), np.array(left_eye))

    # -----------------------

    # apply cosine rule

    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

    # -----------------------

    return img  # return img anyway


def img_resize(img, target_size=(240, 240)):
    """
    resize image without deformation by adding black pixels
    :param img:
    :param target_size:
    :return:
    """
    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        img = cv2.resize(img, dsize)
        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        # Put the base image in the middle of the padded image
        img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),
                     'constant')

    # ------------------------------------------
    # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    return img


# ---------------------------------------------------

def detect_face(img: Image, target_size=(240, 240), detection_threshold=0.85):
    """
    detect and align face and resize in image
    :param detection_threshold: threshold for face detection model to classify the detected object as face
    :param img face image
    :param target_size:
    :return: aligned and resized face image
    """
    global face_detector
    global cuda_available
    if "face_detector" not in globals():
        weights_path = download_detector_model_weights()
        if torch.cuda.is_available():
            face_detector = RetinaFace(gpu_id=1, model_path=weights_path)
        else:
            face_detector = RetinaFace(model_path=weights_path)

    p_width, p_len = img.size
    if p_width > 1024:
        new_width = 1024
        new_len = (new_width * p_len) // p_width
        img = img.resize(((new_width, new_len)))
    img = np.array(img)

    output = face_detector(
        img)  # list of [ box [list of square coordinates ], landmarks [list of landmarks coordinates] , score ]

    detected_faces_attributes = []
    for i in range(len(output)):
        box, landmarks, score = output[i]
        if score >= detection_threshold:
            detected_faces_attributes.append(output[i])

    detected_faces_images = []
    if len(detected_faces_attributes) == 0:
        # raise Exception("no faces found in the given photo")
        return None

    for face_attributes in detected_faces_attributes:
        box, landmarks, score = face_attributes
        left_eye, right_eye = tuple(landmarks[0]), tuple(landmarks[1])

        left_eye, right_eye = (int(left_eye[0]), int(left_eye[1])), (int(right_eye[0]), int(right_eye[1]))
        x, y, w, h = box
        x, y, w, h = int(x), int(y), int(w), int(h)

        left_eye = left_eye[0] - x, left_eye[1] - y
        right_eye = right_eye[0] - x, right_eye[1] - y
        face = img[y:h, x:w]

        face = alignment_procedure(face, left_eye, right_eye)
        face = img_resize(face, target_size)
        detected_faces_images.append(face)

    return detected_faces_images

def download_detector_model_weights():
    url = "https://drive.google.com/u/1/uc?id=1a_FLk4TxX2NoKJsrP2h50XXE6_2Nm_A1&export=download"
    weights_path = 'models/final_weights/face_detector.pt'
    if not os.path.exists(weights_path):
        # if not os.path.exists("/Users/user/PycharmProjects/FPT code/2023/Summer2023/DPL302m/Multi-Face-Recognize/app/model/model_weights"):
        #     os.mkdir("/Users/user/PycharmProjects/FPT code/2023/Summer2023/DPL302m/Multi-Face-Recognize/app/model/model_weights")
        # if not os.path.exists("/Users/user/PycharmProjects/FPT code/2023/Summer2023/DPL302m/Multi-Face-Recognize/app/model/model_weights/final_weights"):
        #     os.mkdir("/Users/user/PycharmProjects/FPT code/2023/Summer2023/DPL302m/Multi-Face-Recognize/app/model/model_weights/final_weights")
        gdown.download(url, weights_path, quiet=False)
    return weights_path
