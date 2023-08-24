import os
import random
import sys
import time

import numpy as np
from os.path import join as join_pth
from PIL import Image
import torch
from torch.utils.data import  IterableDataset
from torchvision.transforms.functional import normalize
import sys
sys.path.append('/Users/user/PycharmProjects/FPT code/2023/Summer2023/DPL302m/Multi-Face-Recognize/src/convfacenet_train')  # Add the directory to the module search path
from convfacenet_train import utils


def load_image(path, transform=None, expand=False):
    img = Image.open(path)
    if transform is not None:
        img = transform(img)
        if expand:
            img = img.unsqueeze(0)
    else:
        img = np.array(img)
    return img


def get_pic_features_dict(dataset_pth, model, transform=None, cuda=False, batch_size=1, img_paths=None):
    """
    get features dictionary from dataset_path
    dataset expected path root/x/img1.jpg
    :param dataset_pth: dataset path root
    :param model: descriptor model object
    :param transform: torchvision transform
    :param cuda: (bool) cuda available
    :param batch_size:
    :param img_paths: specify subset from the data
    :return: features dictionary {x/imgname.jpg:[img_feature_vector]}
    """
    cnt = 0.0
    pic_features_dict = {}
    names = os.listdir(dataset_pth)
    model.eval()
    if cuda:
        model.cuda()
    total_time_taken = 0.0
    if img_paths is None:
        img_paths = []
        for name in names:
            folder_pth = join_pth(dataset_pth, name)
            pics_list = os.listdir(folder_pth)
            for pic_name in pics_list:
                pic_path = f"{name}/{pic_name}"
                img_paths.append(pic_path)
    total_photos = len(img_paths)

    with torch.no_grad():
        for s in range(0, total_photos, batch_size):
            ts = time.time()
            e = min(s + batch_size, total_photos)
            imgs = torch.tensor([])
            for i in range(s, e):
                try:
                    img = load_image(dataset_pth + "/" + img_paths[i], transform, expand=True)
                except:
                    pic_features_dict[img_paths[i]] = None
                    continue
                imgs = torch.cat((imgs, img), dim=0)

            if cuda:
                imgs = imgs.cuda()
                features_vectors = model(imgs).cpu().tolist()
            else:
                features_vectors = model(imgs).tolist()
            fet_idx = 0
            for path_idx in range(s, e):
                pic_features_dict[img_paths[path_idx]] = features_vectors[fet_idx]
                fet_idx += 1
            cnt += float(batch_size)
            te = time.time()
            total_time_taken += (te - ts)
            avg_time_per_name = total_time_taken / cnt
            finished_out_of_10 = int((cnt * 10.0) / total_photos)
            remaining_out_of_10 = 10 - finished_out_of_10

            sys.stdout.flush()
            sys.stdout.write("\r data processed [" + str('=' * finished_out_of_10) + str(
                '.' * remaining_out_of_10) + "] time remaing=" + str(avg_time_per_name * (total_photos - cnt) / 60.0)[
                                                                 0:5])
    print()
    sys.stdout.flush()
    model.train()
    return pic_features_dict


def get_imgs_dict(dataset_pth, all_names=None) -> dict:
    """
    load all images in the dataset in a dictionary ,key is the image path relative to dataset path("person/image_name")
     and value is the image in numpy
    :param dataset_pth: the root path of the dataset
    :return: dict {img_path:numpy_img}
    """
    ts = time.time()
    images_dict = {}
    if all_names is None:
        all_names = os.listdir(dataset_pth)
    cnt = 0.0
    total = float(len(all_names))
    for name in all_names:
        photos = os.listdir(join_pth(dataset_pth, name))
        for photo in photos:
            img_pth = join_pth(dataset_pth, name, photo)
            try:
                img = np.array(Image.open(img_pth))
                images_dict['{}/{}'.format(name, photo)] = img
            except:
                print("error loading --> " + img_pth)
        cnt += 1.0
        sys.stdout.flush()
        sys.stdout.write("\r " + str((cnt * 100.0) / total)[:8] + " % of the folders processed")
    te = time.time()
    print()
    sys.stdout.flush()

    print(f" img dict loaded in {round((te - ts) / 60.0, 2)} m")
    return images_dict


class FacesDataset(IterableDataset):
    def __init__(self, dataset_path, no_of_rows, transform=None, load_imgs_from_dict=False, subset=None):
        """
        :param dataset_path: path that has folder of identities and each identity has it's photos
        :param no_of_rows: limit of no of triplet rows you wish to generate (anchor_img,postive_img,negative_img)
        :param subset:(percentage from 0.0 to 1.0) specify a percentage you wish to take from (identities) not all dataset
        :param transform: (torch.transforms)
        :param load_imgs_from_dict:(boolean) load all images from  {"identity/img_name":img} dictionary
        :param img_features_dict: dictionary of {"identity/img_name":img_features} to select hard negative photo
        :param select_from_negative_cnt: no of randomly chosen images you wish to have to select the hardest negative photo
        """
        names_list = os.listdir(dataset_path)
        random.shuffle(names_list)
        if subset is not None:
            names_list = random.sample(names_list, int(subset * len(names_list)))
        self.person_imgs_list = []  # [(name, [list of photos])... ]
        for name in names_list:
            imgs = []
            for img_name in os.listdir(join_pth(dataset_path, name)):
                imgs.append(img_name)
            self.person_imgs_list.append((name, imgs))
        self.no_of_rows = no_of_rows
        self.transform = transform
        self.dataset_path = dataset_path

        self.load_imgs_from_dict = load_imgs_from_dict
        if load_imgs_from_dict:
            self.images_dict = get_imgs_dict(dataset_path, names_list)
        self.pics_used = {}  # dict of pic_relative_path:usage count

    def load_imgs_dict(self):
        self.images_dict = get_imgs_dict(self.dataset_path, )

    def get_random_neg_face(self, anchor_name):
        negative_name, negative_imgs = random.choice(self.person_imgs_list)
        while negative_name == anchor_name:
            negative_name, negative_imgs = random.choice(self.person_imgs_list)
        return f"{negative_name}/{random.choice(negative_imgs)}"

    def load_img(self, img_path):
        if not self.load_imgs_from_dict or img_path not in self.images_dict:
            img = Image.open(self.dataset_path + "/" + img_path)
        else:
            img = self.images_dict[img_path]

        if self.transform is not None:
            img = self.transform(img)
        return img

    def print_usage_statistics(self):
        s = 0.0
        print()
        no_pics = len(self.pics_used)
        for v in self.pics_used.values():
            s += v
        mean = (s / no_pics)
        std = 0.0
        for v in self.pics_used.values():
            std += ((v - mean) ** 2)
        std /= no_pics
        print(
            f"mean of unique pictures usage count = {round(mean, 3)} and std = {round(std, 3)} , no of unique pictures used ={no_pics}")

    def __len__(self):
        return self.no_of_rows

    def add_to_img_used_pics(self, img_path):
        if img_path in self.pics_used:
            self.pics_used[img_path] += 1
        else:
            self.pics_used[img_path] = 1


class FaceHardSelectionDataset(FacesDataset):
    def __init__(self, model, select_from_negative_cnt, cuda, dataset_path, no_of_rows, transform=None,
                 load_imgs_from_dict=False, img_features_dict=None,
                 subset=None):
        super().__init__(dataset_path, no_of_rows, transform, load_imgs_from_dict, subset)

        self.model = model
        self.cuda = cuda
        if img_features_dict is None:
            img_features_dict = get_pic_features_dict(dataset_path, model, transform, cuda, batch_size=10)

        self.img_features_dict = img_features_dict
        self.select_from_negative_cnt = select_from_negative_cnt
        self.curr_epoch_used_pics = set()

    def get_min_dist_face(self, anchor_name, anchor_img_name):
        """
        get negative image which is closer to the positive image
        :param anchor_name:
        :param anchor_img_name:
        :return: img_path for the chosen image
        """
        imgs_path = []
        anchor_vector = self.img_features_dict[f"{anchor_name}/{anchor_img_name}"]
        for i in range(self.select_from_negative_cnt):
            person_name, person_imgs = random.choice(self.person_imgs_list)
            rand_img = random.choice(person_imgs)
            while person_name == anchor_name or (f"{person_name}/{rand_img}" in self.curr_epoch_used_pics):
                person_name, person_imgs = random.choice(self.person_imgs_list)
                rand_img = random.choice(person_imgs)
            imgs_path.append(f"{person_name}/{rand_img}")
        min_dist = np.Inf
        min_img_path = ""
        for img_path in imgs_path:
            negative_vector = self.img_features_dict[img_path]
            dist = utils.euclidean_distance(anchor_vector, negative_vector)
            if dist < min_dist:
                min_img_path = img_path
                min_dist = dist
        return min_img_path

    def add_to_img_used_pics(self, img_path):
        super().add_to_img_used_pics(img_path)
        self.curr_epoch_used_pics.add(img_path)

    def update_features_dict(self):
        used_pics_features = get_pic_features_dict(self.dataset_path, self.model, self.transform, self.cuda,
                                                   batch_size=10, img_paths=list(self.curr_epoch_used_pics))
        self.curr_epoch_used_pics = set()
        for key, item in used_pics_features.items():
            self.img_features_dict[key] = item


class FacesTripletDataset(FacesDataset):
    def __init__(self, dataset_path, no_of_rows, transform=None, load_imgs_from_dict=False,
                 subset=None, ):
        super().__init__(dataset_path, no_of_rows, transform, load_imgs_from_dict, subset)

    def __iter__(self):
        random.shuffle(self.person_imgs_list)
        for i in range(self.no_of_rows):
            # 1- select random anchor person
            anchor_per_name, anchor_imgs = random.choice(self.person_imgs_list)
            while len(anchor_imgs) < 2:
                anchor_per_name, anchor_imgs = random.choice(self.person_imgs_list)
            # 2- select two random pictures for the choosen person
            random_two_same_pics = random.sample(anchor_imgs, 2)

            a_img_name = '{}/{}'.format(anchor_per_name, random_two_same_pics[0])
            p_img_name = '{}/{}'.format(anchor_per_name, random_two_same_pics[1])

            # 3- select random negative picture that it's feature close to the chosen person
            n_img_name = self.get_random_neg_face(anchor_per_name)

            a_img = self.load_img(a_img_name)
            p_img = self.load_img(p_img_name)
            n_img = self.load_img(n_img_name)

            self.add_to_img_used_pics(a_img_name)
            self.add_to_img_used_pics(p_img_name)
            self.add_to_img_used_pics(n_img_name)

            yield a_img, p_img, n_img


class FacesHardTripletDataset(FaceHardSelectionDataset):
    def __init__(self, model, cuda, dataset_path, no_of_rows, transform=None, load_imgs_from_dict=False,
                 img_features_dict=None,
                 subset=None,
                 select_from_negative_cnt=0):
        super().__init__(model, select_from_negative_cnt, cuda, dataset_path, no_of_rows, transform,
                         load_imgs_from_dict, img_features_dict, subset)

    def __iter__(self):
        random.shuffle(self.person_imgs_list)
        self.update_features_dict()
        for i in range(self.no_of_rows):
            # 1- select random anchor person
            anchor_per_name, anchor_imgs = random.choice(self.person_imgs_list)
            while len(anchor_imgs) < 2:
                anchor_per_name, anchor_imgs = random.choice(self.person_imgs_list)
            # 2- select two random pictures for the choosen person
            random_two_same_pics = random.sample(anchor_imgs, 2)

            a_img_name = '{}/{}'.format(anchor_per_name, random_two_same_pics[0])
            p_img_name = '{}/{}'.format(anchor_per_name, random_two_same_pics[1])

            # 3- select random negative picture that it's feature close to the chosen person
            n_img_name = self.get_min_dist_face(anchor_per_name, anchor_img_name=random_two_same_pics[0])

            a_img = self.load_img(a_img_name)
            p_img = self.load_img(p_img_name)
            n_img = self.load_img(n_img_name)

            self.add_to_img_used_pics(a_img_name)
            self.add_to_img_used_pics(p_img_name)
            self.add_to_img_used_pics(n_img_name)

            yield a_img, p_img, n_img


class FacesPairDataset(FacesDataset):
    def __init__(self, dataset_path, no_of_rows, transform=None, load_imgs_from_dict=False,
                 subset=None):
        assert no_of_rows % 2 == 0

        super().__init__(dataset_path, no_of_rows, transform, load_imgs_from_dict, subset)

    def __iter__(self):
        cnt = 0
        random.shuffle(self.person_imgs_list)
        while cnt <= self.no_of_rows:

            # 1- select random anchor person
            anchor_per_name, anchor_imgs = random.choice(self.person_imgs_list)
            while len(anchor_imgs) < 2:
                anchor_per_name, anchor_imgs = random.choice(self.person_imgs_list)
            # 2- select two random pictures for the choosen person
            random_two_same_pics = random.sample(anchor_imgs, 2)

            a_img_name = '{}/{}'.format(anchor_per_name, random_two_same_pics[0])
            p_img_name = '{}/{}'.format(anchor_per_name, random_two_same_pics[1])

            # 3- select random negative picture that it's feature close to the chosen person
            n_img_name = self.get_random_neg_face(anchor_per_name)

            a_img = self.load_img(a_img_name)
            p_img = self.load_img(p_img_name)
            n_img = self.load_img(n_img_name)

            self.add_to_img_used_pics(a_img_name)
            self.add_to_img_used_pics(p_img_name)
            self.add_to_img_used_pics(n_img_name)

            yield a_img, p_img, torch.tensor([1.0])
            yield a_img, n_img, torch.tensor([0.0])
            cnt += 2


class Normalize(torch.nn.Module):

    def forward(self, img):
        t_mean = torch.mean(img, dim=[1, 2])
        t_std = torch.std(img, dim=[1, 2])
        return normalize(img, t_mean.__array__(), t_std.__array__())

    def __init__(self):
        super().__init__()
