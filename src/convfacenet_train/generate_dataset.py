import random
import numpy as np
import os
from os.path import join as join_pth


def split_data(dataset_pth, split_ratio=0.2):
    names = os.listdir(dataset_pth)
    random.shuffle(names)
    test_names = random.sample(names, int(len(names) * split_ratio))
    test_names_set = set(test_names)
    names_set = set(names)
    train_names_set = names_set.difference(test_names_set)
    train_names = list(train_names_set)
    return train_names, test_names


def generate_testing_data_set_frame(dataset_pth, a_neg_single_subset=True):
    """
    :param dataset_pth: path to faces dataset root where root has root/person[i]/img[i] ...
    :param a_neg_single_subset: get acnchor negative from people with one photo only
    :return: dataFrame of pairs
    """

    namesList = np.array(os.listdir(dataset_pth))
    np.random.shuffle(namesList)
    all_persons_imgs = []  # [name, [list of photos] ]
    multi_img_persons = []  # [name, [list of photos] ]
    single_img_persons = []  # [name, [one_photo] ]

    for name in namesList:
        imgs = []
        for img_name in os.listdir(join_pth(dataset_pth, name)):
            imgs.append(img_name)
        all_persons_imgs.append([name, imgs])
        if len(imgs) == 1:
            single_img_persons.append([name, imgs])
        else:
            multi_img_persons.append([name, imgs])

    datasetList = []
    # anchor negative is from single photos only
    for person_photos in multi_img_persons:
        name = person_photos[0]
        photos = person_photos[1]
        for i in range(len(photos)):
            # anchor
            img1 = '{}/{}'.format(name, photos[i])
            for j in range(i + 1, len(photos)):
                # row (img1,img2, same=1)
                # anchor postive
                datasetList.append([img1, '{}/{}'.format(name, photos[j]), 1])
                # anchor negative
                if a_neg_single_subset:
                    rand_person = random.choice(single_img_persons)
                    rand_person_name = rand_person[0]
                    rand_photo = rand_person[1][0]
                    datasetList.append([img1, '{}/{}'.format(rand_person_name, rand_photo), 0])
                else:
                    rand_person = random.choice(all_persons_imgs)
                    while rand_person[0] == name:
                        rand_person = random.choice(all_persons_imgs)
                    rand_person_name = rand_person[0]
                    rand_person_photos = rand_person[1]
                    rand_photo = np.random.choice(rand_person_photos)
                    datasetList.append([img1, '{}/{}'.format(rand_person_name, rand_photo), 0])
    return datasetList
