import math
import numpy as np

from .. import utils

from PIL import Image

def alignment_procedure(img, left_eye, right_eye):
    """
    this function aligns given face in img based on left and right eye coordinates
    :param img: np_array
    :param left_eye: (x,y)
    :param right_eye: (x,y)
    :return:
    """
    #

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


def detect_face(img_path, detector, target_size=(240, 240)):
    img = Image.open(img_path)
    p_width, p_len = img.size
    if p_width > 1024:
        new_width = 1024
        new_len = (new_width * p_len) // p_width
        img = img.resize(((new_width, new_len)))
    img = np.array(img)

    output = detector(img)  # box, landmarks, score
    # get max face score
    max_score = 0
    max_idx = 0
    for i in range(len(output)):
        box, landmarks, score = output[i]
        if max_score < score:
            max_score = score
            max_idx = i

    box, landmarks, score = output[max_idx]

    left_eye, right_eye = tuple(landmarks[0]), tuple(landmarks[1])

    left_eye, right_eye = (int(left_eye[0]), int(left_eye[1])), (int(right_eye[0]), int(right_eye[1]))

    # box ---> (x1,y1,x2,y2)
    x1, y1, x2, y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
    left_eye = left_eye[0] - x1, left_eye[1] - y1
    right_eye = right_eye[0] - x1, right_eye[1] - y1
    face = img[y1:y2, x1:x2]

    face = alignment_procedure(face, left_eye, right_eye)
    face = utils.img_resize(face, target_size)

    return face

