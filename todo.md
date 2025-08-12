# 🚀 Custom DRAEM 구현 프로젝트 TODO

## 📋 프로젝트 개요

HDMAP 데이터셋을 위한 커스텀 DRAEM 모델 구현 ✅ **2024년 12월 완료**
- **기반**: 기존 DRAEM + Fault Severity Prediction Sub-Network 추가
- **타겟**: 1채널 그레이스케일 HDMAP 데이터 (256x256)
- **특징**: 직사각형 패치 기반 Synthetic Fault Generation
- **신규 기능**: 확률적 Anomaly Generation (0~100% 조절 가능)
- **성과**: 5가지 Severity 입력 모드, 완전한 테스트 검증

## 🏗️ 구현 단계별 계획

### Phase 1: 기본 구조 구축 ⭐
- [x] **1.1 폴더 구조 생성**
  ```
  src/anomalib/models/image/custom_draem/
  ├── __init__.py
  ├── lightning_model.py
  ├── torch_model.py
  ├── loss.py
  ├── synthetic_generator.py
  └── README.md
  ```

- [x] **1.2 CustomDraemModel PyTorch 모델 구현 완료**
  - ✅ ReconstructiveSubNetwork (1채널 그레이스케일 지원)
  - ✅ DiscriminativeSubNetwork (2채널 입력)
  - ✅ FaultSeveritySubNetwork (다중 입력 모드 지원)
  - ✅ 5가지 Severity 입력 모드 구현:
    - discriminative_only (2채널)
    - with_original (3채널) 
    - with_reconstruction (3채널)
    - with_error_map (3채널)
    - multi_modal (5채널)
  - ✅ 통합 테스트 완료 (모든 모드 검증)
  - ✅ 모델 파라미터: 총 1,894,244개 (~7.2MB)

- [x] **1.3 HDMAPCutPasteSyntheticGenerator 구현 완료**
  - ✅ 직사각형 패치 기반 Cut-Paste 클래스 설계 완료
  - ✅ 사용자 정의 가능한 파라미터들 모두 구현:
    - `patch_width_range`: 패치 너비 범위 (픽셀 단위)
    - `patch_ratio_range`: height/width 비율 (>1.0: portrait, <1.0: landscape, 1.0: square)
    - `severity_max`: 최대 severity 값 (연속값, 기본 10.0)
    - `patch_count`: 패치 개수 (기본값: 1, Multi-patch 지원)
    - `validation_enabled`: 경계 검증 자동화
  - ✅ 출력: synthetic_image, fault_mask, severity_map, severity_label
  - ✅ **핵심 구현 완료**: Cut-Paste 알고리즘, 패치 생성, 마스크 생성 모두 구현
  - ✅ **전체 테스트 통과**: 모든 설정, 멀티패치, 실제 HDMAP 데이터 테스트 완료

- [x] **1.4 기존 DRAEM을 1채널 그레이스케일로 수정 완료**
  - ✅ ReconstructiveSubNetwork: in_channels=1, out_channels=1
  - ✅ DiscriminativeSubNetwork: in_channels=2 (original + reconstruction)

### Phase 2: Severity Prediction Sub-Network 구현 ⭐⭐

- [x] **2.1 FaultSeveritySubNetwork 설계 완료**
  - ✅ **Ablation Study를 위한 입력 조합 설계**:
    - `discriminative_only`: Discriminative result만 (2채널) - **baseline**
    - `with_original`: discriminative + original (3채널) - 원본 정보 효과
    - `with_reconstruction`: discriminative + reconstruction (3채널) - 복원 품질 정보
    - `with_error_map`: discriminative + |orig-recon| (3채널) - 에러 명시적 정보
    - `multi_modal`: discriminative + original + reconstruction + error (5채널) - **최대 정보**
  - ✅ **효율적 Ablation을 위한 설계**:
    - 모듈화된 입력 처리: 동적 채널 수 조정
    - 동일한 backbone 네트워크에서 입력 채널 수만 변경
    - 실험 스크립트에서 간단한 설정 변경으로 모든 조합 테스트 가능

- [x] **2.2 네트워크 아키텍처 구현 완료**
  ```python
  # ✅ 구현된 구조: CNN + Global Pooling + MLP
  feature_extractor: Conv2d → BatchNorm → ReLU (3 layers)
  global_pooling: AdaptiveAvgPool2d(1)
  regressor: MLP (3 layers with dropout)
  output: Continuous severity value (0 ~ severity_max)
  ```
  - ✅ 동적 입력 채널 수 처리 (2~5채널)
  - ✅ Feature extraction + Global pooling + Regression
  - ✅ 모든 입력 모드에서 테스트 완료

- [x] **2.3 CustomDraemLoss 구현 완료**
  - ✅ Multi-task loss: reconstruction + segmentation + severity
  - ✅ **Loss 가중치 설정** (사용자 설정 가능):
    - `reconstruction_weight`: 1.0 (기본값, L2 + SSIM 포함)
    - `segmentation_weight`: 1.0 (기본값, Focal loss)
    - `severity_weight`: 0.5 (기본값, 보조 task로 설정)
  - ✅ Severity loss: MSELoss 또는 SmoothL1Loss 선택 가능
  - ✅ 개별 loss 컴포넌트 분석 기능 (`get_individual_losses()`) 포함

### Phase 3: 모델 통합 ⭐⭐⭐ ✅ 완료

- [x] **3.1 CustomDraemModel 구현 완료**
  ```python
  # ✅ 구현 완료
  class CustomDraemModel(nn.Module):
      def __init__(self, 
                   severity_max: float = 10.0,
                   severity_input_mode: str = "discriminative_only"):
          # ✅ 3개 서브네트워크 통합 완료
          # ✅ 5가지 실험 가능한 입력 모드 지원
  ```
  - ✅ Training/Inference 모드 모두 지원
  - ✅ 모든 Severity 입력 모드 통합 테스트 완료
  - ✅ 모델 파라미터 최적화 (총 1.89M 파라미터)

- [x] **3.2 CustomDraem Lightning 모델 구현 완료**
  - ✅ AnomalibModule 상속 구조 완성
  - ✅ 모든 컴포넌트 초기화 완료 (Model + Loss + Generator)
  - ✅ training_step() 구현 완료
  - ✅ validation_step() 구현 완료
  - ✅ configure_optimizers() 완성
  - ✅ 모든 에러 수정 및 shape 불일치 해결

- [x] **3.3 전처리 파이프라인 통합**
  - ✅ 256x256 이미지 처리 완료
  - ✅ 1채널 그레이스케일 처리 완료
  - ✅ 정규화 로직 통합

### Phase 4: 통합 테스트 및 검증 ⭐⭐⭐⭐ ✅ 완료

- [x] **4.1 기본 기능 테스트**
  - ✅ 가상 데이터로 모든 severity 모드 테스트
  - ✅ Training/Validation step 정상 동작 확인
  - ✅ Loss 계산 및 출력 shape 검증

- [x] **4.2 실제 HDMAP 데이터 통합 테스트**
  - ✅ 실제 HDMAP 데이터 로딩 및 배치 생성
  - ✅ Train/Test 시나리오 구현 (train/good → test/good+fault)
  - ✅ 실제 이미지에서 모델 동작 검증
  - ✅ Anomaly detection 파이프라인 확인

- [x] **4.3 End-to-End 파이프라인 검증**
  - ✅ Synthetic generation → Model training → Inference 전체 플로우
  - ✅ 모든 에러 해결 및 안정성 확인
  - ✅ 실제 데이터 호환성 검증

### Phase 5: 확률적 Synthetic Generation 개선 ⭐⭐ ✅ 완료

- [x] **5.1 기존 DRAEM 확률 로직 분석 완료**
  - ✅ PerlinAnomalyGenerator probability=0.5 (50:50 비율) 확인
  - ✅ 사용자 조절 가능 파라미터 분석
  - ✅ Current Custom DRAEM 100% anomaly 생성 문제점 파악

- [x] **5.2 HDMAPCutPasteSyntheticGenerator 확률 로직 추가 완료**
  - ✅ probability 파라미터 추가 (기본값: 0.5)
  - ✅ 확률적 anomaly 생성 로직 구현
  - ✅ Normal 이미지 반환 로직 추가 (빈 마스크 + 원본 이미지)
  - ✅ 파라미터 검증 (0.0 ≤ probability ≤ 1.0)

- [x] **5.3 CustomDraem Lightning 모델 업데이트 완료**
  - ✅ anomaly_probability 파라미터 추가
  - ✅ Generator 초기화에 probability 전달
  - ✅ Docstring 업데이트
  - ✅ 하위 호환성 유지 (기본값 0.5)

- [x] **5.4 확률적 생성 테스트 완료**
  - ✅ 다양한 probability 값 테스트 (0.0, 0.3, 0.5, 0.7, 1.0)
  - ✅ Normal vs Anomaly 비율 검증 (최대 오차 3.5%)
  - ✅ Lightning 모델 통합 테스트 (0.3, 0.5, 0.8)
  - ✅ 모든 테스트 케이스 통과

## 📊 실험 설정

### Severity Network 입력 모드 실험
1. **discriminative_only**: 기본 모드 (Discriminative result만)
2. **with_original**: + Original image
3. **with_reconstruction**: + Reconstruction  
4. **with_error_map**: + Reconstruction error map
5. **multi_modal**: 모든 정보 조합

### Synthetic Fault Generation 파라미터
- **patch_ratio_range**: 
  - Landscape: (2.0, 4.0) - 가로로 긴 직사각형 (HDMAP 특성 반영)
  - Portrait: (0.25, 0.5) - 세로로 긴 직사각형 (실험용)
  - Square: 1.0 - 정사각형 패치 (비교군)
- **patch_size_range**: (20, 80) - **256x256 리사이즈 후 픽셀 기준**
  - 최소: 20픽셀 (약 8% of image)
  - 최대: 80픽셀 (약 31% of image)
  - **자동 검증**: 이미지 경계 초과 방지
- **severity_max**: 10.0 - 실험적으로 조정 가능 (연속값)
- **patch_count**: **기본 1개**, ablation: 1,2,3
  - **동일 설정 제약**: 다중 패치 시 모든 패치가 같은 severity/ratio/size

### 모델 하이퍼파라미터
- **이미지 크기**: 256x256 (전처리로 통일)
- **배치 크기**: 16 (GPU 메모리 고려)
- **학습률**: 0.0001 (기존 DRAEM 기준)
- **Loss 가중치**: reconstruction:segmentation:severity = 1.0:1.0:0.5 
  - **Note**: 초기 설정값이며, 실험을 통해 최적 조합 탐색 필요
  - 예상 조합: [1.0:1.0:0.3], [1.0:1.0:0.7], [0.8:1.0:0.5] 등

## 🎯 현재 진행 상황

- [x] ✅ 기존 DRAEM 구조 분석 완료
- [x] ✅ Custom DRAEM 설계 완료  
- [x] ✅ TODO 문서 작성 완료
- [x] ✅ Phase 1: 기본 구조 구축 완료
- [x] ✅ Phase 2: Severity Prediction Sub-Network 구현 완료
- [x] ✅ Phase 3: 모델 통합 완료 (PyTorch + Lightning)
- [x] ✅ Phase 4: 테스트 및 검증 완료 (가상 + 실제 HDMAP 데이터)
- [x] ✅ **Phase 5: 확률적 Synthetic Generation 개선 완료** **← 2024년 최신 완료**

## 🚀 다음 작업 우선순위 (2024년 진행)

### ✅ 완료: Phase 1-5 모든 핵심 구현
- **Phase 1**: ✅ 기본 구조 구축 (폴더, 모델, 생성기)
- **Phase 2**: ✅ Severity Prediction Sub-Network 구현
- **Phase 3**: ✅ 모델 통합 (PyTorch + Lightning)
- **Phase 4**: ✅ 통합 테스트 및 검증 (가상 + 실제 HDMAP 데이터)
- **Phase 5**: ✅ 확률적 Synthetic Generation 개선 (probability=0.0~1.0)

### 🎯 **2024년 12월 최신 달성**: Custom DRAEM 핵심 구현 100% 완료!

#### 📊 **Phase 5 완료 성과 요약**:
- ✅ **확률적 Anomaly Generation**: 0~100% 조절 가능 (최대 오차 3.5%)
- ✅ **기존 DRAEM 호환**: 동일한 probabilistic generation 방식 적용
- ✅ **완전한 테스트 검증**: 5개 테스트 시나리오 모두 통과
- ✅ **실제 데이터 호환**: HDMAP 데이터셋 완전 지원
- ✅ **하위 호환성**: 기존 API 변경 없이 새 기능 추가

### 🔥 다음 우선순위: 실제 학습 및 성능 평가
- **실험 스크립트 작성**: single/multi-domain HDMAP 학습
- **성능 비교**: Custom DRAEM vs 기존 DRAEM
- **Ablation Study**: Severity network 입력 모드별 성능 분석 (5가지 모드)
- **확률적 학습 실험**: anomaly_probability 0.3, 0.5, 0.8 성능 비교
- **도메인 전이 실험**: 다양한 HDMAP 도메인 간 전이 성능

## 📁 주요 파일 경로

### 구현할 파일들
- `src/anomalib/models/image/custom_draem/` - 메인 모델 구현
- `examples/hdmap/custom_draem_experiments/` - 실험 스크립트

### 참고할 기존 파일들
- `src/anomalib/models/image/draem/` - 기존 DRAEM 구현
- `examples/hdmap/multi_domain_hdmap_*_training.py` - 기존 실험 스크립트
- `src/anomalib/data/datamodules/image/multi_domain_hdmap.py` - 데이터 모듈

## 💡 참고사항

### Train vs Test 단계 처리
- **Training**: Synthetic fault generation → 3개 서브네트워크 학습
- **Testing**: Real fault detection → Severity 예측 (Ground Truth 없이)
- **핵심**: Severity Network는 Discriminative result 기반으로만 예측

### 확장 가능성
- 다양한 패치 모양 (원형, 불규칙형)
- Multi-scale severity prediction
- Domain-adaptive synthetic generation
- Few-shot severity learning

---
**💪 Let's build an awesome Custom DRAEM! 🚀**
