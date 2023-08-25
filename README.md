<h1><p align="center">[PYTORCH] Face Recognition using Convnext Model + Flask</p>
<p align="center">🙋‍♂️ 🙆‍♀️ 🧟 👨‍🦳</p>
</h1>


## Introduction
This repository contains Python source code designed for both training and inference tasks related to person identification. During the training process, we utilize a feature extractor to obtain vectors, which are then compared with our database using a webcam.

To provide context, the Triplet loss serves as a significant aspect in Machine Learning. Its objective is to facilitate the learning of an embedding space where similar instances are positioned closer to each other, while dissimilar instances are placed farther apart. This loss function involves triplets of examples: an anchor, a positive example, and a negative example.

<p align="center">
    <img src="image_demo/demo-running-result.png", width="500"><br/>
    <i> Sample result </i>
</p>

## The database format
Database structure is a list contain many dictionaries, the format be like: 
```
[
    {
        "name": person1,
        "face_feature": face feature person1 embedding
    }
]
```
*Currently, I'm storing data using the .pickle file format, but there are concerns about its safety. In the future, I intend to transition to a different file format for storing the database, such as .npz or safetensors. This change aims to enhance the overall trustworthiness of the storage mechanism.*

## How to use my code

*Since the configuration is currently set to run with the defaults specified in both main.yaml and process1.yaml, you have the option to modify the configuration details in order to achieve the optimal experimental outcome.*

### Manipulating database (Add, Delete, Rename, etc)
    python src/manipulate_data.py

### Inference on app (Flask) using the current trained model: 
    python src/main.py

## Reference
I would like to express my graditude to @ahmedbadr97 for creating the increadible repository [conv-facenet](https://github.com/ahmedbadr97/conv-facenet), from which most of this code is derived. Thank you very much!




