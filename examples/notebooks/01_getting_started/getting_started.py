## import
from typing import Any
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore, Draem, Dsr, Fastflow, Cfa, Stfpm

## 데이터셋 준비
datamodule = MVTecAD(
    num_workers=95,
    train_batch_size=4,    # gpu memory에 따라서 조절
    eval_batch_size=4,     # gpu memory에 따라서 조절
)
datamodule.prepare_data()  # Downloads the dataset if it's not in the specified `root` directory
datamodule.setup()  # Create train/val/test/prediction sets.

i, data = next(enumerate(datamodule.val_dataloader()))
print(type(data))
print(data.image.shape, data.gt_mask.shape)

# ============================================================================
# 모델 선택 - 아래 MODEL_NAME을 수정하여 다른 모델 테스트 가능
# ============================================================================
MODEL_NAME = "Stfpm" 
# 지원 모델: Patchcore, DRAEM, PaDiM, DSR, FastFlow, CFA, Stfpm

# 모델 매핑 딕셔너리
MODEL_MAP = {
    "Patchcore": Patchcore,
    "DRAEM": Draem,
    "PaDiM": Padim,
    "DSR": Dsr,
    "FastFlow": Fastflow,
    "CFA": Cfa,
    "Stfpm": Stfpm,
}

# 모델 생성
if MODEL_NAME not in MODEL_MAP:
    available_models = list(MODEL_MAP.keys())
    raise ValueError(f"지원하지 않는 모델: {MODEL_NAME}. 사용 가능한 모델: {available_models}")

ModelClass = MODEL_MAP[MODEL_NAME]
model = ModelClass()

print(f"선택된 모델: {MODEL_NAME}")

# 모델별 특별 설정
model_specific_settings = {
    "max_epochs": 1,  # 기본값
    "num_sanity_val_steps": 2,
}

# Patchcore와 PaDiM은 1 에포크만 필요
if MODEL_NAME in ["Patchcore", "PaDiM"]:
    model_specific_settings["max_epochs"] = 1
    print(f"ℹ{MODEL_NAME}은 특징 추출 기반 모델로 1 에포크만 실행됩니다.")
else:
    model_specific_settings["max_epochs"] = 5  # 다른 모델들은 더 많은 에포크
    print(f"ℹ{MODEL_NAME}은 {model_specific_settings['max_epochs']} 에포크 훈련됩니다.")

## GPU 설정
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "12"  # 12번 GPU만 사용하도록 환경 변수 설정

# GPU 메모리 정리 및 최적화 설정
import torch
import gc

# GPU 캐시 정리
torch.cuda.empty_cache()
gc.collect()

print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU 메모리 예약량: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


from anomalib.loggers import AnomalibTensorBoardLogger
from lightning.pytorch.callbacks import Callback
import lightning.pytorch as pl

# 콜백과 로거 설정
experiment_name = f"my_experiment_{MODEL_NAME.lower()}"
tensorboard_logger = AnomalibTensorBoardLogger(
    save_dir="logs",
    name=experiment_name
)
print(f"TensorBoard 실험명: {experiment_name}")

engine = Engine(
    accelerator="gpu", 
    devices=[0],
    logger=tensorboard_logger,                    # TensorBoard 로거 설정
    max_epochs=model_specific_settings["max_epochs"],  # 모델에 따라 동적 설정
    check_val_every_n_epoch=1,                    # 매 에포크마다 validation 실행
    enable_checkpointing=True,                    # 체크포인트 활성화 (메트릭 저장에 필요)
    log_every_n_steps=1,                          # 매 스텝마다 로깅
    enable_model_summary=True,                    # 모델 요약 활성화
    num_sanity_val_steps=model_specific_settings["num_sanity_val_steps"],  # validation sanity check
)
engine.fit(model=model, datamodule=datamodule)

# load best model from checkpoint before evaluating
test_results = engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
)

print("\n" + "="*70)
print(f"{MODEL_NAME} 모델 학습 및 테스트 완료!")
print("="*70)