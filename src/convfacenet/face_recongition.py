import os
import gdown as gdown
import torch.cuda
from torch import no_grad

import sys
sys.path.append('/home/pphuc/Downloads/testing-code/Multi-Face-Recognize/Multi-Face-Recognize')
from src.convfacenet.face_detection import *
from src.convfacenet.face_descriptor.models import FaceDescriptorModel
from torchvision import transforms


def load_image(img_path):
    return Image.open(img_path)


def faces_features(img: Image):
    """
    extract faces features in a image
    :param img: PIL image
    :return: list of [numpy face features vector dim=128]
    """
    is_face = False
    faces_images = detect_face(img, target_size=(240, 240), detection_threshold=0.5)
    if faces_images is None or len(faces_images) == 0:
        # raise ValueError("No faces detected in the passed photo")
        return is_face, None
    global face_descriptor
    global model_transform
    is_face = True
    cuda_available = torch.cuda.is_available()
    if "face_descriptor" not in globals() or face_descriptor is None:
        face_descriptor = FaceDescriptorModel()
        load_descriptor_model_weights(face_descriptor)
        if cuda_available:
            face_descriptor.cuda()

    if "model_transform" not in globals():
        model_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    face_descriptor.eval()
    faces_features_list = []
    for face_image in faces_images:
        face_transformed = model_transform(face_image)
        face_transformed = face_transformed.unsqueeze(0)
        if cuda_available:
            face_transformed = face_transformed.cuda()
        with no_grad():
            face_features = face_descriptor(face_transformed)[0]
            if cuda_available:
                face_features = face_features.cpu()
            faces_features_list.append(face_features.numpy())
    return is_face, faces_features_list


def verify_faces(face1_img, face2_img, threshold=0.5):
    """
    verify that two faces are for same person
    :param face1_path: first picture path
    :param face2_path: second picture path
    :param threshold: threshold of verification from 0 to 1 0 means they are identical
    :return: tuple (boolean value verified or not , distance between faces from 0 to 1)
    """
    face_1_features = faces_features(face1_img)[0]
    face_2_features = faces_features(face2_img)[0]
    cos_distance = utils.findCosineDistance(face_1_features, face_2_features)
    if cos_distance > threshold:
        return False, cos_distance
    else:
        return True, cos_distance


def load_descriptor_model_weights(model):
    url = "https://drive.google.com/u/1/uc?id=1sighXzyFufqurh4M4dqB4sNWcTA9PccO&export=download"
    weights_path = "models/final_weights/face_descriptor.pt"
    if not os.path.exists(weights_path):
    #     if not os.path.exists(
    #             "/Users/user/PycharmProjects/FPT code/2023/Summer2023/DPL302m/Multi-Face-Recognize/app/model/model_weights"):
    #         os.mkdir(
    #             "/Users/user/PycharmProjects/FPT code/2023/Summer2023/DPL302m/Multi-Face-Recognize/app/model/model_weights")
    #     if not os.path.exists(
    #             "/Users/user/PycharmProjects/FPT code/2023/Summer2023/DPL302m/Multi-Face-Recognize/app/model/model_weights/final_weights"):
    #         os.mkdir(
    #             "/Users/user/PycharmProjects/FPT code/2023/Summer2023/DPL302m/Multi-Face-Recognize/app/model/model_weights/final_weights")
        gdown.download(url, weights_path, quiet=False)
    model.load_local_weights(weights_path, True)
