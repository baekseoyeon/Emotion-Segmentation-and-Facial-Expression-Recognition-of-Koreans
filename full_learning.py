import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import KFold
from collections import defaultdict, Counter

# 데이터 전처리 및 변환
data_transforms = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(20),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 전체 데이터셋 로드
data_dir = '/content/dataset/crop'  # 클래스 폴더들이 있는 경로
image_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

# 인물 ID 추출 함수
def extract_person_id(file_path):
    filename = os.path.basename(file_path)
    person_id = filename.split('_')[0]
    return person_id

# 클래스별 ID 수 출력 함수
def print_class_id_distribution(dataset):
    class_id_map = defaultdict(set)
    for img_path, target in dataset.samples:
        person_id = extract_person_id(img_path)
        class_id_map[dataset.classes[target]].add(person_id)
    
    print("Class ID distribution:")
    for class_name, ids in class_id_map.items():
        print(f"Class '{class_name}': {len(ids)} unique IDs")

# 클래스별 이미지 수 계산 함수
def count_class_distribution(idxs, dataset):
    class_distribution = Counter(dataset.targets[idx] for idx in idxs)
    return class_distribution

# K-Fold 분할 및 클래스별 이미지 수 출력 함수
def print_fold_distribution(splits, dataset):
    for fold_idx, (train_ids, valid_ids) in enumerate(splits):
        print(f'Fold {fold_idx+1}')
        
        train_distribution = count_class_distribution(train_ids, dataset)
        valid_distribution = count_class_distribution(valid_ids, dataset)
        
        print("Training set class distribution:")
        for class_idx, count in train_distribution.items():
            class_name = dataset.classes[class_idx]
            print(f"Class '{class_name}': {count} images")
        
        print("Validation set class distribution:")
        for class_idx, count in valid_distribution.items():
            class_name = dataset.classes[class_idx]
            print(f"Class '{class_name}': {count} images")
        print('-' * 20)

# 인물 ID 기반 데이터 분할을 위한 함수
def get_person_splits(dataset, n_splits):
    person_ids = list(set(extract_person_id(img_path) for img_path, _ in dataset.samples))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    person_idx_map = defaultdict(list)
    for idx, (img_path, _) in enumerate(dataset.samples):
        person_id = extract_person_id(img_path)
        person_idx_map[person_id].append(idx)
    
    splits = []
    for train_persons, valid_persons in kf.split(person_ids):
        train_idx = [idx for person_id in [person_ids[i] for i in train_persons] for idx in person_idx_map[person_id]]
        valid_idx = [idx for person_id in [person_ids[i] for i in valid_persons] for idx in person_idx_map[person_id]]
        splits.append((train_idx, valid_idx))
    
    return splits

# 클래스별 샘플링을 위한 함수
def get_class_weights(dataset, idxs):
    labels = [dataset.targets[idx] for idx in idxs]
    class_counts = Counter(labels)
    total_samples = len(idxs)
    weights = [total_samples / (len(class_counts) * class_counts[label]) for label in labels]
    return weights

# 모델 정의 (드롭아웃 추가)
class MyResNet(nn.Module):
    def __init__(self, original_model):
        super(MyResNet, self).__init__()
        self.resnet = original_model
        self.dropout = nn.Dropout(p=0.1)  # 드롭아웃 비율 설정
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)  # 드롭아웃 적용
        return x

# 모델 학습 함수
def train_model_kfold(model, criterion, optimizer, scheduler, num_epochs=10):
    best_acc = 0.0
    splits = get_person_splits(image_dataset, k_folds)

    print_class_id_distribution(image_dataset)
    print_fold_distribution(splits, image_dataset)
    
    for fold, (train_ids, valid_ids) in enumerate(splits):
        print(f'Fold {fold+1}/{k_folds}')
        print('-' * 10)

        train_subsampler = Subset(image_dataset, train_ids)
        valid_subsampler = Subset(image_dataset, valid_ids)

        print("Training set class ID distribution:")
        print_class_id_distribution(train_subsampler.dataset)
        print("Validation set class ID distribution:")
        print_class_id_distribution(valid_subsampler.dataset)

        train_weights = get_class_weights(image_dataset, train_ids)
        train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
        
        train_loader = DataLoader(train_subsampler, batch_size=64, sampler=train_sampler, num_workers=0)
        valid_loader = DataLoader(valid_subsampler, batch_size=64, shuffle=False, num_workers=0)
        
        dataloaders = {'train': train_loader, 'valid': valid_loader}

        best_model_wts = model.state_dict()
        patience_counter = 0
        best_epoch_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                dataset_size = len(dataloaders[phase].dataset)

                for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_size
                epoch_acc = running_corrects.double() / dataset_size

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'valid':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = model.state_dict()
                        patience_counter = 0
                    else:
                        patience_counter += 1
            
            scheduler.step()
            
            if patience_counter >= 5:
                print("Early stopping due to no improvement")
                break

    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # 모델 초기화
    original_model = models.resnet18(weights='DEFAULT')
    num_ftrs = original_model.fc.in_features
    original_model.fc = nn.Linear(num_ftrs, 7)
    model = MyResNet(original_model)

    # 손실 함수 및 옵티마이저 설정
    class_counts = Counter(image_dataset.targets)
    total_samples = len(image_dataset.targets)
    class_weights = [total_samples / (len(class_counts) * count) for count in class_counts.values()]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)  # L2 정규화 추가
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    k_folds = 5  # 원하는 K-Fold 수 설정
    model = train_model_kfold(model, criterion, optimizer, scheduler, num_epochs=5)

    torch.save(model.state_dict(), 'emotion_resnet18_kfold.pth')
++++++++++++++++++++


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# 데이터 로드 시 ToTensor로 변환
data_transforms = transforms.Compose([
    transforms.ToTensor(),
])

# 데이터셋 로드
dataset = datasets.ImageFolder('/content/dataset/crop', transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

# 평균과 표준편차 계산 함수
def calculate_mean_std(loader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images, _ in loader:
        # 이미지 개수
        batch_size = images.size(0)
        total_images += batch_size
        
        # 각 채널에 대해 평균 계산 (배치 기준)
        for i in range(3):
            mean[i] += images[:, i, :, :].mean() * batch_size
            std[i] += images[:, i, :, :].std() * batch_size

    mean /= total_images
    std /= total_images
    
    return mean, std

# 평균과 표준편차 계산
mean, std = calculate_mean_std(dataloader)

print(f'Mean: {mean}')
print(f'Std: {std}')
