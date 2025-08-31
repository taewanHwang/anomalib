# Single Domain Base Template 설계 문서

## 🎯 목표
각 모델별로 개별 파일을 만드는 대신, 공통 베이스 템플릿을 사용하여 모든 anomaly detection 모델의 단일 도메인 실험을 통합 관리

## 📋 현재 문제점 분석

### 반복되는 코드들 (공통 부분)
1. **데이터 로딩**: HDMAPDataModule 생성 로직
2. **실험 관리**: timestamp, 경로 설정, 로깅
3. **콜백 설정**: EarlyStopping, ModelCheckpoint
4. **결과 저장**: JSON 결과 파일 생성
5. **메트릭 설정**: test_image_AUROC 설정
6. **훈련 루프**: trainer.fit(), trainer.test() 호출

### 모델별 차이점 (개별 부분)
1. **모델 생성**: 각 모델의 고유 파라미터
2. **옵티마이저 설정**: 모델별 최적화 전략
3. **ValidationLoss 콜백**: DRAEM 계열은 수동 val_loss 계산 필요

## 🏗️ 제안하는 통합 구조

### 1. Base Training Script (`base-training.py`)
```python
class BaseAnomalyTrainer:
    def __init__(self, config, experiment_name, session_timestamp):
        self.model_type = config["model_type"]  # "draem", "dinomaly", "patchcore" 등
        # 공통 초기화 로직
    
    def create_model(self):
        # Factory pattern으로 모델 타입별 생성
        if self.model_type == "draem":
            return self._create_draem_model()
        elif self.model_type == "dinomaly": 
            return self._create_dinomaly_model()
        # ...
    
    def _create_draem_model(self):
        # DRAEM 모델별 설정
    
    def _create_dinomaly_model(self):
        # Dinomaly 모델별 설정
    
    # 공통 메서드들
    def create_datamodule(self):
        # 모든 모델 동일
    
    def create_callbacks(self):
        # 모든 모델에서 val_loss 사용 (single domain에서 validation은 모두 정상 데이터)
    
    def save_results(self):
        # 모든 모델 동일
```

### 2. Base Experiment Conditions (`base-exp_condition.json`)
```json
{
  "experiment_conditions": [
    {
      "name": "domainA_draem_baseline",
      "description": "Domain A - DRAEM 기본 설정",
      "config": {
        "model_type": "draem",           // 새로 추가
        "source_domain": "domain_A",
        "max_epochs": 50,
        "early_stopping_patience": 10,
        "learning_rate": 0.0001,
        "batch_size": 16,
        "image_size": "224x224"
        // 모델별 고유 파라미터들
      }
    },
    {
      "name": "domainA_dinomaly_baseline", 
      "description": "Domain A - Dinomaly 기본 설정",
      "config": {
        "model_type": "dinomaly",        // 새로 추가
        "source_domain": "domain_A",
        "max_epochs": 50,
        "early_stopping_patience": 10,
        "learning_rate": 0.0001,
        "batch_size": 8,
        "image_size": "224x224",
        "encoder_name": "dinov2reg_vit_base_14",
        "target_layers": [2, 3, 4, 5, 6, 7, 8, 9]
        // Dinomaly 고유 파라미터들
      }
    }
  ]
}
```

### 3. Base Run Script (`base-run.sh`)
```bash
#!/bin/bash
# 단일 스크립트로 모든 모델 실험 실행
./base-training.py --config base-exp_condition.json
```

## 🔧 구현 계획

### Phase 1: 기본 구조 구현
1. **BaseAnomalyTrainer 클래스** 작성
   - Factory pattern으로 모델 생성
   - 공통 로직 통합 (데이터, 콜백, 결과 저장)

2. **모델별 Factory 메서드** 구현
   - `_create_draem_model()`
   - `_create_dinomaly_model()`  
   - `_create_patchcore_model()`
   - `_create_draem_sevnet_model()`

3. **통합 실험 조건 JSON** 작성
   - 각 모델의 대표적인 설정 포함
   - model_type 필드 추가

### Phase 2: 고급 기능
1. **Early Stopping 전략 통합**
   - 모든 모델: val_loss 사용 (single domain에서 validation은 모든 정상 데이터이므로 AUROC 부적절)
   - DRAEM 계열: ValidationLossCallback으로 수동 val_loss 계산

2. **옵티마이저 팩토리** 구현
   - 모델별 최적 옵티마이저 자동 선택

3. **결과 분석 도구** 통합
   - 한 번의 실행으로 모든 모델 결과 비교

## 📊 예상 효과

### 장점
1. **코드 중복 제거**: 80% 이상의 공통 코드 통합
2. **실험 일관성**: 모든 모델에 동일한 실험 프레임워크 적용
3. **유지보수성**: 한 곳에서 공통 로직 수정
4. **확장성**: 새로운 모델 추가 시 Factory 메서드만 추가

### 고려사항
1. **모델별 특수 요구사항**: 일부 모델의 고유 로직 처리 필요
2. **설정 복잡성**: JSON에서 모델별 파라미터 관리
3. **디버깅**: 통합된 구조에서 모델별 이슈 추적

## 🚀 다음 단계

1. **현재 Dinomaly pickle 에러 해결**
2. **base-training.py 구현** - Factory pattern 적용
3. **base-exp_condition.json 작성** - 4개 모델 통합
4. **base-run.sh 작성** - 단순화된 실행 스크립트
5. **테스트 및 검증** - 기존 개별 스크립트와 결과 비교

이 통합 접근법으로 **단일 코드베이스**로 모든 anomaly detection 모델의 단일 도메인 실험을 효율적으로 관리할 수 있을 것입니다.