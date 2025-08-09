# mnist_pytorch_basic_debug.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MNIST 분류 (기본 PyTorch 스크립트, 디버그 출력 포함)
- 데이터 개수/shape 확인
- 모델 구조 출력
- dummy 데이터로 forward shape 추적
"""

import os
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

# -----------------------------
# 0) 하드코딩 파라미터
# -----------------------------
DATA_DIR = "./datasets"
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
HIDDEN_DIM = 128
SEED = 42
OUT_DIR = "./results"

# -----------------------------
# 1) 재현성 & 디바이스
# -----------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:15") if torch.cuda.is_available() else torch.device("cpu")

if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

# -----------------------------
# 2) 데이터 준비 + 정보 출력
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

mnist_train_full = MNIST(root=DATA_DIR, train=True, transform=transform, download=True)
mnist_test = MNIST(root=DATA_DIR, train=False, transform=transform, download=True)

generator = torch.Generator().manual_seed(SEED)
mnist_train, mnist_val = random_split(mnist_train_full, [55000, 5000], generator=generator)

# 데이터 개수 출력
print(f"훈련 데이터 개수: {len(mnist_train)}")
print(f"검증 데이터 개수: {len(mnist_val)}")
print(f"테스트 데이터 개수: {len(mnist_test)}")

# 샘플 하나의 shape 출력
sample_img, sample_label = mnist_train[0]
print(f"샘플 이미지 shape: {sample_img.shape}, 라벨: {sample_label}")

train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(mnist_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(mnist_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# -----------------------------
# 3) 모델 & 옵티마이저 + 구조 출력
# -----------------------------
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, HIDDEN_DIM),
    nn.ReLU(),
    nn.Linear(HIDDEN_DIM, 10)
).to(device)

print("\n=== 모델 구조 ===")
print(model)

# Dummy tensor로 shape 변화 추적
print("\n=== Dummy Tensor Forward Shape Trace ===")
x = torch.randn(1, 1, 28, 28).to(device)
print(f"입력: {x.shape}")
for layer in model:
    x = layer(x)
    print(f"{layer.__class__.__name__} 출력: {x.shape}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# -----------------------------
# 4) 학습 & 검증 루프
# -----------------------------
for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss_sum, train_correct, train_count = 0.0, 0, 0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        train_loss_sum += loss.item() * batch_size
        train_correct += (logits.argmax(dim=1) == targets).sum().item()
        train_count += batch_size

    train_loss = train_loss_sum / train_count
    train_acc = train_correct / train_count

    # Validate
    model.eval()
    val_loss_sum, val_correct, val_count = 0.0, 0, 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, targets)

            batch_size = targets.size(0)
            val_loss_sum += loss.item() * batch_size
            val_correct += (logits.argmax(dim=1) == targets).sum().item()
            val_count += batch_size

    val_loss = val_loss_sum / val_count
    val_acc = val_correct / val_count

    print(f"[Epoch {epoch:02d}] "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

# -----------------------------
# 5) 테스트
# -----------------------------
model.eval()
test_loss_sum, test_correct, test_count = 0.0, 0, 0
with torch.no_grad():
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, targets)

        batch_size = targets.size(0)
        test_loss_sum += loss.item() * batch_size
        test_correct += (logits.argmax(dim=1) == targets).sum().item()
        test_count += batch_size

test_loss = test_loss_sum / test_count
test_acc = test_correct / test_count
print(f"[Test] loss={test_loss:.4f} acc={test_acc:.4f}")

# -----------------------------
# 6) 모델 저장
# -----------------------------
os.makedirs(OUT_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(OUT_DIR, "last.pth"))
print(f"Saved weights to {os.path.join(OUT_DIR, 'last.pth')}")
