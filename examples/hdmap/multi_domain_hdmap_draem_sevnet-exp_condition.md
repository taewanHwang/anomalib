# DRAEM-SevNet 실험 설계 가이드

## 실험 파일 개요

### 📂 `multi_domain_hdmap_draem_sevnet-exp_condition1.json`
**실험 목표**: 다양한 패치 형태와 크기 조합을 통한 fault augmentation 최적화  
**실험 개수**: 14개 실험으로 landscape, portrait, square 패치의 다양한 크기 조합 테스트

### 📂 `multi_domain_hdmap_draem_sevnet-exp_condition2.json`
**실험 목표**: severity_max 값 변화에 따른 성능 영향 분석  
**실험 개수**: 16개 실험으로 다양한 severity_max 값 (0.5, 1.0, 2.0, 5.0, 10.0)과 패치 설정 조합 테스트

### 📂 `multi_domain_hdmap_draem_sevnet-exp_condition3.json`
**실험 목표**: severity_head_mode별 성능 차이 비교 (single_scale vs multi_scale)  
**실험 개수**: 16개 실험으로 각 mode별 다양한 패치 설정 조합 테스트

### 📂 `multi_domain_hdmap_draem_sevnet-exp_condition4.json`
**실험 목표**: score_combination 방법별 성능 차이 비교 (simple_average vs weighted_average vs maximum)  
**실험 개수**: 16개 실험으로 각 combination 방법별 다양한 패치 설정 조합 테스트

### 📂 `multi_domain_hdmap_draem_sevnet-exp_condition5.json`
**실험 목표**: severity_loss_type별 성능 차이 및 severity_weight 조합 최적화  
**실험 개수**: 16개 실험으로 다양한 loss function과 weight 조합 테스트 (mse, smooth_l1 + weight 0.5~2.0)

### 📂 `multi_domain_hdmap_draem_sevnet-exp_condition6.json`
**실험 목표**: patch_count 개수별 성능 차이 및 multiple patch 전략 최적화  
**실험 개수**: 16개 실험으로 다양한 patch 개수(1~4개)와 multiple patch 배치 전략 테스트

## 공통 파라미터

모든 실험 파일에서 사용되는 공통 파라미터들:

- **severity_max**: severity 값의 최대 범위 (기본값: 1.0, condition2에서는 0.5~10.0 범위 테스트)
- **severity_weight**: loss function에서 severity loss의 가중치
- **severity_head_mode**: single_scale 또는 multi_scale
- **score_combination**: simple_average, weighted_average, maximum
- **severity_loss_type**: mse 또는 smooth_l1
- **patch_width_range**: 패치 너비 범위
- **patch_ratio_range**: 패치 종횡비 범위  
- **patch_count**: 패치 개수

## 사용법

실험을 실행하려면 `multi_domain_hdmap_draem_sevnet-training.py` 스크립트에서 원하는 실험 파일을 로드하세요:

```python
# 예시: condition3 실험 실행
EXPERIMENT_CONDITIONS = load_experiment_conditions("multi_domain_hdmap_draem_sevnet-exp_condition3.json")
```
