# from ultralytics import YOLO
# import cv2
# import os
# from PIL import Image

# # YOLOv8-face 모델 로드
# model = YOLO('yolov8n-face.pt')

# # 입력 폴더 설정
# input_folder = 'C:/Users/seoye/Desktop/한국인 감정인식을 위한 복합 영상/Training'

# padding_ratio = 0

# # 사람별로 처리된 이미지 수를 저장할 딕셔너리
# person_image_count = {}

# # 폴더 순회
# for subfolder in os.listdir(input_folder):
#     subfolder_path = os.path.join(input_folder, subfolder)

#     # '한국인 감정인식을 위한 복합 영상' 폴더 건너뛰기
#     if subfolder == '[원천]EMOIMG_분노_TRAIN_02':
#         print(f"Skipping folder: {subfolder}")
#         continue

#     # '[원천]EMOIMG_기쁨_TRAIN_02' 폴더 건너뛰기
#     if subfolder == '[원천]EMOIMG_기쁨_TRAIN_02':
#         print(f"Skipping folder: {subfolder}")
#         continue

#     # '[원천]EMOIMG_기쁨_TRAIN_02' 폴더 건너뛰기
#     if subfolder == '[원천]EMOIMG_당황_TRAIN_02':
#         print(f"Skipping folder: {subfolder}")
#         continue

#         # '[원천]EMOIMG_기쁨_TRAIN_02' 폴더 건너뛰기
#     if subfolder == '[원천]EMOIMG_불안_TRAIN_02':
#         print(f"Skipping folder: {subfolder}")
#         continue

#             # '[원천]EMOIMG_기쁨_TRAIN_02' 폴더 건너뛰기
#     if subfolder == '[원천]EMOIMG_상처_TRAIN_02':
#         print(f"Skipping folder: {subfolder}")
#         continue

#                 # '[원천]EMOIMG_기쁨_TRAIN_02' 폴더 건너뛰기
#     if subfolder == '[원천]EMOIMG_슬픔_TRAIN_02':
#         print(f"Skipping folder: {subfolder}")
#         continue


#     if os.path.isdir(subfolder_path):
#         # 출력 폴더 생성 (입력 폴더의 하위 폴더 이름을 기준으로)
#         output_folder = os.path.join(input_folder, 'cropssssss', subfolder)
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)
        
#         for file_name in os.listdir(subfolder_path):
#             file_path = os.path.join(subfolder_path, file_name)
            
#             # 파일 이름에서 사람 ID 추출 (예: '9d16b5061afdad0ad1326fa3cdc5ca6eb62a4e35d34361d95485f32c86be6d32' 형식)
#             person_id = file_name.split('_')[0]
            
#             # 사람 ID별로 이미지 수 확인, 15개 초과 시 처리하지 않고 다음 사람으로 넘어감
#             if person_id not in person_image_count:
#                 person_image_count[person_id] = 0
#             if person_image_count[person_id] >= 1:
#                 print(f"Skipping {file_name}: Already processed 15 images for person {person_id}")
#                 continue
            
#             if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 # 이미지 로드 및 예측
#                 results = model(file_path)
                
#                 # 얼굴 바운딩 박스 추출
#                 boxes = results[0].boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표를 numpy 배열로 변환
                
#                 # 얼굴 영역 추출 및 저장
#                 img = cv2.imread(file_path)
#                 img_height, img_width = img.shape[:2]

#                 # 각 바운딩 박스 순회 (첫 번째 얼굴만 저장)
#                 if len(boxes) > 0:
#                     box = boxes[0]  # 첫 번째 바운딩 박스만 사용
#                     if len(box) == 4:  # 바운딩 박스가 4개의 요소를 가진 배열인지 확인
#                         x1, y1, x2, y2 = map(int, box)  # 좌표를 정수형으로 변환

#                         # 바운딩 박스에 여백 추가
#                         width = x2 - x1
#                         height = y2 - y1
#                         x1 = max(int(x1 - width * padding_ratio), 0)
#                         y1 = max(int(y1 - height * padding_ratio), 0)
#                         x2 = min(int(x2 + width * padding_ratio), img_width)
#                         y2 = min(int(y2 + height * padding_ratio), img_height)

#                         # 얼굴 부분 추출
#                         face_crop = img[y1:y2, x1:x2]
                        
#                         # 얼굴 부분 이미지로 변환
#                         face_crop_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                        
#                         # 이미지 리사이즈 (224x224)
#                         face_crop_resized = face_crop_img.resize((224, 224), Image.Resampling.LANCZOS)
                        
#                         # 저장 경로 설정 (원래 파일명 유지)
#                         output_path = os.path.join(output_folder, file_name)
                        
#                         # 이미지 저장
#                         face_crop_resized.save(output_path)
#                         print(f"Saved cropped face image to {output_path}")
                        
#                         # 사람 ID별로 이미지 수 증가
#                         person_image_count[person_id] += 1
#                     else:
#                         print(f"Unexpected bbox format: {box}")

# print("Processing complete.")



from ultralytics import YOLO
import cv2
import os
from PIL import Image

# YOLOv8-face 모델 로드
model = YOLO('yolov8n-face.pt')

# 입력 폴더 설정
input_folder = 'C:/Users/seoye/Desktop/한국인 감정인식을 위한 복합 영상/Training'

padding_ratio = 0

# 사람별로 처리된 이미지 수를 저장할 딕셔너리
person_image_count = {}

# 폴더 순회
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

    if os.path.isdir(subfolder_path):
        # 출력 폴더 생성 (입력 폴더의 하위 폴더 이름을 기준으로)
        output_folder = os.path.join(input_folder, 'cropssssss', subfolder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        for file_name in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file_name)
            
            # 파일 이름에서 사람 ID 추출
            person_id = file_name.split('_')[0]
            
            # 사람 ID별로 이미지 수 확인, 12개 초과 시 처리하지 않고 다음 사람으로 넘어감
            if person_id not in person_image_count:
                person_image_count[person_id] = 0
            if person_image_count[person_id] >= 10:
                print(f"Skipping {file_name}: Already processed 12 images for person {person_id}")
                continue
            
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 이미지 로드 및 예측
                results = model(file_path)
                
                # 얼굴 바운딩 박스 추출
                boxes = results[0].boxes.xyxy.cpu().numpy()
                
                # 얼굴 영역 추출 및 저장
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
                        
                        # 얼굴 부분 이미지로 변환
                        face_crop_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                        
                        # 이미지 리사이즈 (224x224)
                        face_crop_resized = face_crop_img.resize((224, 224), Image.Resampling.LANCZOS)
                        
                        # 저장 경로 설정
                        output_path = os.path.join(output_folder, file_name)
                        
                        # 이미지 저장
                        face_crop_resized.save(output_path)
                        print(f"Saved cropped face image to {output_path}")
                        
                        # 사람 ID별로 이미지 수 증가
                        person_image_count[person_id] += 1
                    else:
                        print(f"Unexpected bbox format: {box}")

print("Processing complete.")
