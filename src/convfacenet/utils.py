import cv2
import numpy as np
import json
import shutil
import os



def get_last_weights_path(weights_path):
    files = os.listdir(weights_path)
    files.sort()
    last_weight_file_name = files[-1]
    return f"{weights_path}/{last_weight_file_name}"


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def euclidean_distance(vector_a, vector_b):
    if type(vector_a) != "numpy.ndarray":
        vector_a = np.array(vector_a)
    if type(vector_b) != "numpy.ndarray":
        vector_b = np.array(vector_b)

    diff = vector_a - vector_b
    # sum squared
    sum_squared = np.dot(diff.T, diff)
    return np.sqrt(sum_squared)


def save_dict_to_json(path_file, data_dict):
    with open(path_file, 'w') as fp:
        json.dump(data_dict, fp)


def copydir(src, dest):
    shutil.copytree(src, dest + '/' + src.split('/')[-1], copy_function=shutil.copy)


def copyfile_to_dir(fpath, dirPath):
    shutil.copyfile(fpath, dirPath + '/' + fpath.split('/')[-1])


def load_dict_from_json(path_file):
    with open(path_file) as json_file:
        data = json.load(json_file)
    return data


def img_resize(img, target_size):
    """
    resize image to expected shape
    :param img:
    :param target_size: (width,length)
    :return:
    """
    # ---------------------------------------------------
    # resize image to expected shape
    # img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image

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
