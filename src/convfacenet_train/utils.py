import numpy as np
import json
import shutil
import os
def get_last_weights_path(weights_path):
    files=os.listdir(weights_path)
    files.sort()
    last_weight_file_name=files[-1]
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
