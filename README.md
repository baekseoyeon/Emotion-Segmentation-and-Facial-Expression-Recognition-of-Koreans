# Emotion-Segmentation-and-Facial-Expression-Recognition-of-Koreans 😀🥲🤨😏
## Introduction

Emotion recognition technology is highly versatile and plays a critical role in enhancing human-machine interactions. 

In particular, the Korean  society, influenced by Confucian values, often suppresses emotional expressions or contains complex emotions. 

Thus, continuous research into emotion recognition for Koreans is essential. 

This research, based on the emotion classification framework of Susan David, uses a Korean facial expression dataset to develop a model that can accurately recognize Korean emotional expressions. 

The study, presented in the 2024 Korean Internet Information Society Annual Conference Proceedings, Volume 25, Issue 2, aims to build a more accurate emotion recognition model for Koreans using YOLOv8-Face and ResNet-18. 

The model was trained and evaluated to recognize seven emotions (joy, sadness, surprise, anger, anxiety, hurt, and neutral) while minimizing data imbalance through preprocessing. 

With the optimal hyperparameters, the final model achieved a test accuracy of 91.89% with precision of 0.9135, recall of 0.9187, and F1-score of 0.9151. 

In comparison with the previous research on Korean facial expression detection, which achieved 84.37% accuracy with 49,000 images, this study reached 91.89% with just 35,000 images—showing a 7.52% improvement in accuracy.

## Repository Structure
`preprocess.py`: Preprocesses raw images using the YOLOv8-face model to detect and crop faces.

`classfication.py`: Splits the dataset into training, validation, and test sets based on specified ratios.

`cross-validation.py`: Implements a 5-fold cross-validation pipeline using ResNet-18 for emotion classification.

## Setup

### 1. Environment Configuration
Ensure you have Python 3.8 or later installed.

To run the code in this repository, install the following dependencies:

```bash
pip install torch torchvision scikit-learn ultralytics opencv-python Pillow
```

### 2. Run the Code
(1) Preprocessing Images

The `preprocess.py` script uses the YOLOv8-face model to detect faces in the dataset and crop them for further processing.

Set the input folder path in `preprocess.py`:
```bash
input_folder = 'Emotion-Segmentation-and-Facial-Expression-Recognition-of-Koreans/data/train'
```

Run the script:
```bash
python preprocess.py
```

(2) Splitting Dataset

The `classfication.py` script splits the dataset into training, validation, and test sets.

Update the dataset directory path in `classfication.py`:
```bash
data_dir = 'Emotion-Segmentation-and-Facial-Expression-Recognition-of-Koreans/data'
```

Run the script:
```bash
python classfication.py
```
* Currently, the dataset in `Emotion-Segmentation-and-Facial-Expression-Recognition-of-Koreans/data/` has already been split.

(3) Training with Cross-Validation

The `cross-validation.py` is a script that trains an emotion classification model using ResNet-18 with 5-Fold Cross-Validation.

Run the script:
```bash
python cross-validation.py
```

⚠ Note:
The dataset must be properly defined as image_datasets['train'].

Depending on the torchvision version, the usage of models.resnet18(weights=...) may vary.

## Output
Cropped face images are saved in a subdirectory under the input folder.

Trained model weights for each fold are saved as `emotion_resnet18.pth`.

## Notice
The YOLOv8-face model file (yolov8n-face.pt) must be downloaded and placed in the working directory before running `preprocess.py`.

Modify batch size, learning rate, or other hyperparameters in `cross-validation.py` as needed.

You can obtain the raw dataset from AI Hub. For more details, please refer to the paper.
