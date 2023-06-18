import os
from os.path import join as join_pth
import pandas as pd


def get_name_photos_list(dataset_path):
    """
    returns list of person and no of photos per person
    :return: [[person,no_of_photos]]
    """
    name_photos_list = []
    names = os.listdir(dataset_path)
    for name in names:
        name_photos_list.append([name, len(os.listdir(os.path.join(dataset_path, name)))])
    return pd.DataFrame(name_photos_list, columns=["identity", "no_photos"])


def remove_unknown_files(dataset_path):
    ##Remove Txt files from Folders
    cnt_removed = 0
    allNames = os.listdir(dataset_path)
    for name in allNames:
        files = os.listdir(join_pth(dataset_path, name))
        imgFullPath = ""
        for fname in files:
            if os.path.splitext(fname)[-1] != '.jpg':
                imgFullPath = name + '/' + fname
                os.remove(join_pth(dataset_path, imgFullPath))
                cnt_removed += 1
    return cnt_removed
