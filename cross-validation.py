import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 데이터셋 인덱스
dataset = image_datasets['train']
dataset_size = len(dataset)

# 교차 검증
for fold, (train_idx, valid_idx) in enumerate(kf.split(range(dataset_size))):
    print(f'Fold {fold+1}')
    
    train_subset = Subset(dataset, train_idx)
    valid_subset = Subset(dataset, valid_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_subset, batch_size=32, shuffle=False, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # torchvision 0.13 이상 호환
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train_model(model, {'train': train_loader, 'valid': valid_loader}, criterion, optimizer, num_epochs=10)
    torch.save(model.state_dict(), f'emotion_resnet18_fold{fold+1}.pth')
