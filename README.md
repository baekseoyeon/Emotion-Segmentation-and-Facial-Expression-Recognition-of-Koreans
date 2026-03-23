# Emotion Segmentation and Facial Expression Recognition of Koreans

한국인 얼굴 표정 이미지 기반 감정 분류 실험을 정리한 연구 저장소입니다.  
이 레포는 얼굴 전처리, 데이터 분할, 분류 모델 학습 흐름을 중심으로 구성되어 있습니다.

## 프로젝트 개요

이 프로젝트는 한국인 얼굴 표정 이미지를 대상으로 감정을 분류하기 위한 실험을 정리한 저장소입니다.

주요 작업은 다음과 같습니다.

- YOLOv8-face 기반 얼굴 검출 및 crop 전처리
- 학습/검증/테스트 데이터 분할
- ResNet-18 기반 감정 분류 실험

이 레포는 서비스 배포용 프로젝트라기보다,  
연구 실험 과정과 코드 흐름을 정리한 아카이브 성격의 저장소입니다.

## 이 레포에서 다루는 범위

- 입력 데이터: 한국인 얼굴 표정 이미지
- 전처리: 얼굴 검출 및 crop, 224x224 리사이즈
- 분류 대상: 7개 감정
  - 기쁨
  - 슬픔
  - 당황
  - 분노
  - 불안
  - 상처
  - 중립
- 모델
  - YOLOv8-face
  - ResNet-18

## 저장소 구성

```bash
.
├── README.md
├── preprocess.py
├── classfication.py
├── cross-validation.py
├── 2024년도 한국인터넷정보학회 추계학술발표대회 논문집 제25권 2호.pdf
└── docs/
    └── FILE_GUIDE.md
```

## 파일 설명

##### `preprocess.py`

원본 이미지에서 얼굴 영역을 검출하고 crop한 뒤, 224x224 크기로 저장하는 전처리 스크립트입니다.  
YOLOv8-face를 사용합니다.

##### `classfication.py`

클래스별 이미지 폴더를 train / valid / test로 분할하는 스크립트입니다.  
현재는 8:1:1 비율의 단순 분할 방식으로 구성되어 있습니다.

##### `cross-validation.py`

ResNet-18 기반 감정 분류 모델을 5-fold cross-validation으로 학습하는 실험 스크립트입니다.

##### `발표 논문`

본 연구가 수록된 학술대회 논문집 파일입니다.  

## 실험 흐름

### 1. 얼굴 전처리

`preprocess.py`에서 얼굴을 검출하고 crop하여 학습 가능한 입력 형태로 변환합니다.

### 2. 데이터 분할

`classfication.py`에서 감정 클래스별 이미지를 학습/검증/테스트 세트로 분할합니다.

### 3. 모델 학습

`cross-validation.py`에서 ResNet-18 기반 감정 분류 모델을 5-fold 방식으로 학습합니다.

## 실행 환경

권장 환경은 다음과 같습니다.

- Python 3.8 이상
- PyTorch
- torchvision
- scikit-learn
- ultralytics
- opencv-python
- Pillow

설치 예시는 아래와 같습니다.

```bash
pip install torch torchvision scikit-learn ultralytics opencv-python Pillow
```

## 참고 사항

- 일부 스크립트는 로컬 경로와 실험 환경을 기준으로 작성되어 있습니다.
- 전체 학습 과정은 추가 설정이 있어야 바로 재현될 수 있습니다.
- 이 저장소는 완전한 배포형 프로젝트보다는 연구 실험 흐름과 코드 구조를 정리하는 목적에 가깝습니다.

## 논문 자료

연구 결과는 아래 논문집 파일에서 확인할 수 있습니다.

- [발표 논문`](./2024년도%20한국인터넷정보학회%20추계학술발표대회%20논문집%20제25권%202호.pdf)

논문을 더 선명하게 보여주기 위해, 레포에는 코드뿐 아니라 결과물 확인용 논문집 파일도 함께 포함했습니다.

## 문서 안내

구체적인 파일별 역할은 아래 문서에도 정리되어 있습니다.

- [`docs/FILE_GUIDE.md`](./docs/FILE_GUIDE.md)
