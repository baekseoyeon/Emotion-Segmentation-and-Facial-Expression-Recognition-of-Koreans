import os
import shutil
import random

# 데이터셋 경로
data_dir = 'Emotion-Segmentation-and-Facial-Expression-Recognition-of-Koreans/data/'

# Train/Valid/Test 비율 설정
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

# 클래스별 폴더 순회
for emotion in os.listdir(data_dir):
    emotion_path = os.path.join(data_dir, emotion)
    if os.path.isdir(emotion_path):
        images = os.listdir(emotion_path)
        random.shuffle(images)

        train_split = int(len(images) * train_ratio)
        valid_split = int(len(images) * (train_ratio + valid_ratio))

        train_images = images[:train_split]
        valid_images = images[train_split:valid_split]
        test_images = images[valid_split:]

        for category, img_list in [('train', train_images), ('valid', valid_images), ('test', test_images)]:
            output_dir = os.path.join(data_dir, category, emotion)
            os.makedirs(output_dir, exist_ok=True)
            for img in img_list:
                src = os.path.join(emotion_path, img)
                dst = os.path.join(output_dir, img)
                shutil.copy(src, dst)
