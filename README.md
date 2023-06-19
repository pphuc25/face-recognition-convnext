# [PYTORCH] Face Recognition using Convnext Model

## Introduction
This repo is python source for training and inference to recognite who is this person, , we train and get the feature extracter to extract vector and compare with our database via webcame.

For your information, Triplet loss is a loss function used in Machine Learning, the goal of it is to learn an embedding space where similar instances are closer to each other, while dissimilar instances are farther apart. The loss funciton takes into account triplets of examples: an anchor, a positive example, and a negative example

## How to use my code
With my code you can:
- Train the model: In progress...
- Test your trained model (or my model) by running `bash run.sh`


### Step by step:
- Runinng command `git clone https://github.com/pphuc25/face-recognition.git`
- Change direction to folder you have clone: `cd face-recognition`
- Install libraries (Recommend to use env like conda): `pip install -r requirements.txt`

#### To training (In progress...)

#### To inference
- run file run.sh by command `bash run.sh` (If you use Linux)
- Open the IP in comamnd line