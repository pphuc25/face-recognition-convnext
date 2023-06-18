
import torch

from torch import load, save, nn, Tensor
from torchvision.models import convnext_tiny


class FaceDescriptorModel(nn.Module):

    def __init__(self, download_weights=False, output_size=128):
        super().__init__()

        convnext_tiny_model=convnext_tiny(pretrained=download_weights)
        self.features=convnext_tiny_model.features
        # Change Full connected layer
        self.classifier = convnext_tiny_model.classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier[-1]=nn.Linear(self.classifier[-1].in_features,output_size)

    def load_local_weights(self, path, cuda_weights=False):
        if cuda_weights:
            device = torch.device('cpu')
            state_dict = load(path, map_location=device)
        else:
            state_dict = load(path)
        self.load_state_dict(state_dict)


    def save_weights(self, path):

        state_dict = self.state_dict()
        save(state_dict, path)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def feature_vector(self, faces, transform=None):
        """
        calculate 128 feature vector for given image(s)

        :param faces: after transform img must be tensor of size 240x240
        :param transform:
        :return: nx128 tensor feature vector where n is images size
        """
        self.eval()
        shape = faces.shape
        if transform is not None:
            faces = transform(faces)
        if len(shape) == 3:
            faces=faces.unsqueeze(0)
        with torch.no_grad():
            output = self(faces)
        return output


class FacenetClassifier(nn.Module):
    def __init__(self, face_features_dim=128, descriptor_weights_path=None):
        super().__init__()

        self.descriptor = FaceDescriptorModel(False)
        if descriptor_weights_path is not None:
            self.descriptor.load_local_weights(descriptor_weights_path, True)
        self.classifier = nn.Sequential(nn.Linear(face_features_dim * 2, 128), nn.ReLU(inplace=True), nn.Dropout(),
                                        nn.Linear(128, 1),nn.Sigmoid())

    def forward(self, face_x, face_y):
        assert face_x.shape == face_y.shape
        assert len(face_x.shape) == len(face_y.shape) == 4
        face_x_features = self.descriptor(face_x)
        face_y_features = self.descriptor(face_y)
        classifier_input = torch.cat((face_x_features, face_y_features), dim=1)
        output = self.classifier(classifier_input)
        return output

    def load_local_weights(self, path, cuda_weights=False):
        if cuda_weights:
            device = torch.device('cpu')
            state_dict = load(path, map_location=device)
        else:
            state_dict = load(path)
        self.load_state_dict(state_dict)

    def save_weights(self, path):

        state_dict = self.state_dict()
        save(state_dict, path)

    def identify_faces(self, face_x, face_y, transform=None):
        if transform is not None:
            face_x = transform(face_x)
            face_y = transform(face_y)
        self.eval()
        with torch.no_grad():
            output = self.forward(face_x, face_y)
        return output

    def classify_face_features(self, face_x, face_y, cuda=False):
        self.eval()
        if not isinstance(face_x, torch.Tensor):
            face_x = torch.tensor(face_x)
        if not isinstance(face_y, torch.Tensor):
            face_y = torch.tensor(face_y)

        if len(face_x.shape) == 1:
            face_x = face_x.unsqueeze(0)
        if len(face_y.shape) == 1:
            face_y = face_y.unsqueeze(0)
        with torch.no_grad():

            classifier_input = torch.cat((face_x, face_y), dim=1)
            if cuda:
                classifier_input = classifier_input.cuda()
            output = self.classifier(classifier_input)
        if cuda:
            return output[0].cpu().numpy()
        else:
            return output[0].numpy()


