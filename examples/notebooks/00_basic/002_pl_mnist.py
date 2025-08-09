# mnist_lightning_debug.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MNIST 분류 (PyTorch Lightning, 디버그 출력 포함)
- 데이터 개수/shape 확인
- 모델 구조 출력
- dummy 데이터로 forward shape 추적
- 체크포인트/조기종료/로깅(TensorBoard)
"""

import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


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
LOG_DIR = "./logs"

pl.seed_everything(SEED, workers=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "15"

# -----------------------------
# 1) LightningModule (모델/루프)
# -----------------------------
class MyClassifier(pl.LightningModule):
    """
    - training_step/validation_step/test_step에서 cross_entropy, acc 로깅
    - configure_optimizers에서 AdamW 사용
    """
    def __init__(self, hidden_dim: int = 128, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        """
        train/val/test 단계에서 공통으로 쓰이는 로직을 모아둔 함수.
        - 중복 코드를 줄이고, 로직 일관성을 유지하기 위해 자주 사용됨.
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # epoch 단위 집계 로깅
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=(stage != "train"))
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=(stage != "train"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        # 학습률 로깅(예시)
        opt = self.optimizers()
        if opt is not None and len(opt.param_groups) > 0:
            self.log("learning_rate", opt.param_groups[0]["lr"], on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer


# -----------------------------
# 2) LightningDataModule (데이터)
# -----------------------------
class MyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./datasets", batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                full, [55000, 5000], generator=torch.Generator().manual_seed(SEED)
            )
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,   # 병렬 데이터 로딩에 사용할 CPU 프로세스 개수
            pin_memory=True                 # GPU로 데이터 전송 시 속도 향상을 위해 고정 메모리 사용
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


# -----------------------------
# 3) 디버그 출력 (데이터/모델/shape trace)
# -----------------------------
dm = MyDataModule(data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=4)
dm.prepare_data()
dm.setup("fit")
dm.setup("test")

print(f"훈련 데이터 개수: {len(dm.mnist_train)}")
print(f"검증 데이터 개수: {len(dm.mnist_val)}")
print(f"테스트 데이터 개수: {len(dm.mnist_test)}")

sample_img, sample_label = dm.mnist_train[0]
print(f"샘플 이미지 shape: {sample_img.shape}, 라벨: {sample_label}")

model = MyClassifier(hidden_dim=HIDDEN_DIM, lr=LR)
print("\n=== 모델 구조 ===")
print(model)

print("\n=== Dummy Tensor Forward Shape Trace ===")
x = torch.randn(1, 1, 28, 28)
print(f"입력: {x.shape}")
for layer in model.model:
    x = layer(x)
    print(f"{layer.__class__.__name__} 출력: {x.shape}")

# -----------------------------
# 4) 로거/콜백/트레이너
# -----------------------------
os.makedirs(OUT_DIR, exist_ok=True)
logger = TensorBoardLogger(save_dir=LOG_DIR, name="mnist_mlp")

ckpt = ModelCheckpoint(
    dirpath=OUT_DIR,
    filename="mnist-mlp-{epoch:02d}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)
early = EarlyStopping(monitor="val_loss", patience=3, mode="min")

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator="gpu",
    devices=1,
    callbacks=[ckpt, early],
    logger=logger,
    deterministic=True,
    log_every_n_steps=50,
)

# -----------------------------
# 5) 학습 & 테스트
# -----------------------------
trainer.fit(model, datamodule=dm)
print(f"Best checkpoint: {ckpt.best_model_path} (score={ckpt.best_model_score})")

trainer.test(model, datamodule=dm, ckpt_path=ckpt.best_model_path or "best")
print("Done.")

# TensorBoard로 로그 보기:
# !tensorboard --logdir ./tb_logs