# Emotion-Segmentation-and-Facial-Expression-Recognition-of-Koreans 😀🥲🤨😏
## Introduction

<p align="center">
  <img src="image.png" width="400">
</p>

Emotion recognition technology is highly versatile and plays a critical role in enhancing human-machine interactions. 

In particular, the Korean  society, influenced by Confucian values, often suppresses emotional expressions or contains complex emotions. 

Thus, continuous research into emotion recognition for Koreans is essential. 

This research, based on the emotion classification framework of Susan David, uses a Korean facial expression dataset to develop a model that can accurately recognize Korean emotional expressions. 

The study, presented in the 2024 Korean Internet Information Society Annual Conference Proceedings, Volume 25, Issue 2, aims to build a more accurate emotion recognition model for Koreans using YOLOv8-Face and ResNet-18. 

The model was trained and evaluated to recognize seven emotions (joy, sadness, surprise, anger, anxiety, hurt, and neutral) while minimizing data imbalance through preprocessing. 

With the optimal hyperparameters, the final model achieved a test accuracy of 91.89% with precision of 0.9135, recall of 0.9187, and F1-score of 0.9151. 

In comparison with the previous research on Korean facial expression detection, which achieved 84.37% accuracy with 49,000 images, this study reached 91.89% with just 35,000 images—showing a 7.52% improvement in accuracy.

## Repository Structure
preprocess.py: Preprocesses raw images using the YOLOv8-face model to detect and crop faces.
classfication.py: Splits the dataset into training, validation, and test sets based on specified ratios.
cross-validation.py: Implements a 5-fold cross-validation pipeline using ResNet-18 for emotion classification.

## Setup

### 1. Environment Configuration
Ensure you have Python 3.8 or later installed.

To run the code in this repository, install the following dependencies:

```bash
pip install torch torchvision scikit-learn ultralytics opencv-python Pillow
```

### 2. Run the Code
(1) Preprocessing Images
The preprocess.py script uses the YOLOv8-face model to detect faces in the dataset and crop them for further processing.
Set the input folder path in preprocess.py:
```bash
input_folder = 'Training'
```

(2) Data Processing
You can preprocess the data using chunking.py or sentences.py:
```bash
python chunking.py
python sentences.py
```

(3) Run the Web Application
To launch the Streamlit-based web application, execute:
```bash
streamlit run alphastorm_app.py
```

## Data Notice
This project uses private data provided by contest organizers. The dataset is not included in this repository. 

Users must prepare and place the data manually.
