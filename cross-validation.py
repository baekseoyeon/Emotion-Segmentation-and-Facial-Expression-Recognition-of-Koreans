from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler

# KFold 객체 생성 (K=5)
kf = KFold(n_splits=5, shuffle=True)

# 데이터셋 인덱스 가져오기
dataset = image_datasets['train']
dataset_size = len(dataset)

# 교차 검증 실행
for fold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold+1}')
    
    # 각 fold의 학습 및 검증 데이터 설정
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler, num_workers=0)
    valid_loader = DataLoader(dataset, batch_size=32, sampler=valid_sampler, num_workers=0)
    
    # 모델 학습
    model = models.resnet18(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 기존 학습 함수에 교차 검증 데이터로 학습 실행
    model = train_model(model, {'train': train_loader, 'valid': valid_loader}, criterion, optimizer, num_epochs=10)
    
    # 모델 저장 등 추가 작업
    torch.save(model.state_dict(), f'emotion_resnet18_fold{fold+1}.pth')
