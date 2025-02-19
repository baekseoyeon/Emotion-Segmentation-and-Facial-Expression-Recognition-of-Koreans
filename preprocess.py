from ultralytics import YOLO
import cv2
import os
from PIL import Image

# YOLOv8-face 모델 로드
model = YOLO('yolov8n-face.pt')

input_folder = 'Emotion-Segmentation-and-Facial-Expression-Recognition-of-Koreans/data/train'

padding_ratio = 0
person_image_count = {}

for subfolder in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder)

    # '한국인 감정인식을 위한 복합 영상' 폴더 건너뛰기
    if subfolder == '[원천]EMOIMG_분노_TRAIN_02':
        print(f"Skipping folder: {subfolder}")
        continue

    if subfolder == '[원천]EMOIMG_기쁨_TRAIN_02':
        print(f"Skipping folder: {subfolder}")
        continue

    if subfolder == '[원천]EMOIMG_당황_TRAIN_02':
        print(f"Skipping folder: {subfolder}")
        continue

    if subfolder == '[원천]EMOIMG_불안_TRAIN_02':
        print(f"Skipping folder: {subfolder}")
        continue

    if subfolder == '[원천]EMOIMG_상처_TRAIN_02':
        print(f"Skipping folder: {subfolder}")
        continue

    if subfolder == '[원천]EMOIMG_슬픔_TRAIN_02':
        print(f"Skipping folder: {subfolder}")
        continue
    # 실험 진행 시 압축 풀기로 넣어놓은 폴더 'cropssssss', 이미지 용량이 커 재실험을 대비하여 '한국인 감정인식을 위한 복합 영상' 폴더와 같은 디렉토리 내에 위치하도록 함
    if os.path.isdir(subfolder_path):
        output_folder = os.path.join(input_folder, 'cropssssss', subfolder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        for file_name in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file_name)
            
            person_id = file_name.split('_')[0]
            
            # 사람 ID별로 이미지 수 확인, 12개 초과 시 처리하지 않고 다음 사람으로 넘어감
            if person_id not in person_image_count:
                person_image_count[person_id] = 0
            if person_image_count[person_id] >= 10:
                print(f"Skipping {file_name}: Already processed 12 images for person {person_id}")
                continue
            
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                results = model(file_path)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                img = cv2.imread(file_path)
                img_height, img_width = img.shape[:2]

                # 첫 번째 얼굴만 저장
                if len(boxes) > 0:
                    box = boxes[0]  # 첫 번째 바운딩 박스만 사용
                    if len(box) == 4:
                        x1, y1, x2, y2 = map(int, box)

                        # 바운딩 박스에 여백 추가
                        width = x2 - x1
                        height = y2 - y1
                        x1 = max(int(x1 - width * padding_ratio), 0)
                        y1 = max(int(y1 - height * padding_ratio), 0)
                        x2 = min(int(x2 + width * padding_ratio), img_width)
                        y2 = min(int(y2 + height * padding_ratio), img_height)

                        # 얼굴 부분 추출
                        face_crop = img[y1:y2, x1:x2]
                        face_crop_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                        face_crop_resized = face_crop_img.resize((224, 224), Image.Resampling.LANCZOS)
                        output_path = os.path.join(output_folder, file_name)
                        
                        face_crop_resized.save(output_path)
                        print(f"Saved cropped face image to {output_path}")
                        
                        person_image_count[person_id] += 1
                    else:
                        print(f"Unexpected bbox format: {box}")
print("Processing complete.")
