# Adaptive Bottleneck Dropout for Dinomaly

## 1. 문제 배경

### 관찰된 현상
| Dataset | 200 steps | 1,000 steps | 5,000 steps |
|---------|-----------|-------------|-------------|
| PNG domain_A | 93.12% | 83.44% | 79.42% |
| FFT domain_A | 94.47% | 97.16% | 66.78% |

- Step 증가 시 AUROC **하락** → Overfitting 발생
- 모델이 정상 패턴을 과도하게 학습하여 anomaly contrast collapse 발생

### 원인 분석
- HDMAP 정상 샘플: 강한 규칙적 구조 (dominant orientations, repetitive grids)
- 장기 학습 시 decoder가 "normal pattern generator"로 수렴
- 미세한 anomaly 신호도 정상으로 재구성됨

---

## 2. 해결 방안: Adaptive Bottleneck Dropout

### 핵심 아이디어
**Orientation Entropy** 기반으로 샘플별 dropout 확률 조정:
- **낮은 entropy** (규칙적 패턴) → **높은 dropout** (과적합 방지)
- **높은 entropy** (복잡한 패턴) → **낮은 dropout** (학습 안정성)

### EDA 결과 (PNG Dataset)
```
Normal samples:   entropy = 0.427 (규칙적)
Abnormal samples: entropy = 0.450 (덜 규칙적)
차이: 0.023 → Normal이 더 규칙적임을 확인
```

---

## 3. 구현 파일

### 핵심 모듈
```
src/anomalib/models/image/dinomaly/
├── adaptive_dropout.py          # Orientation entropy 계산 및 adaptive dropout
├── torch_model_adaptive.py      # DinomalyModelAdaptive (torch 모델)
├── lightning_model_adaptive.py  # DinomalyAdaptive (Lightning 래퍼)
└── __init__.py                  # Export 추가됨
```

### EDA 스크립트
```
examples/hdmap/EDA/
├── orientation_entropy.py           # Entropy 계산 함수
├── analyze_orientation_entropy.py   # 분석 스크립트
└── results/orientation_entropy/     # 분석 결과
```

### 검증 스크립트
```
examples/notebooks/
└── hdmap_adaptive_validation.py     # 실험 스크립트
```

---

## 4. 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `base_dropout` | 0.3 | 기준 dropout (sensitivity=0일 때 또는 normal_entropy에서) |
| `min_dropout` | 0.1 | 최소 dropout (normal보다 불규칙한 샘플) |
| `max_dropout` | 0.6 | 최대 dropout (normal보다 규칙적인 샘플) |
| `dropout_sensitivity` | 4.0 | entropy→dropout 매핑의 민감도 (0이면 adaptive 비활성) |
| `normal_entropy` | 0.43 | 정상 샘플의 평균 entropy (HDMAP PNG 기준) |

### Dropout 매핑 공식 (New: tanh-based, centered on normal_entropy)
```python
# deviation: 정상 데이터 entropy와의 차이
deviation = normal_entropy - entropy
# 양수: 정상보다 규칙적 → dropout 증가
# 음수: 정상보다 불규칙적 → dropout 감소

delta = deviation * sensitivity
adjustment = tanh(delta)  # Range: [-1, 1]

# 비대칭 매핑 (base_dropout 중심)
if adjustment >= 0:
    dropout = base_dropout + adjustment * (max_dropout - base_dropout)
else:
    dropout = base_dropout + adjustment * (base_dropout - min_dropout)
```

**핵심 특성:**
- `sensitivity=0` → `dropout = base_dropout` (기존 Dinomaly와 동일)
- `entropy = normal_entropy` → `dropout = base_dropout`
- `entropy < normal_entropy` → dropout 증가 (과적합 방지)
- `entropy > normal_entropy` → dropout 감소 (학습 안정성)

---

## 5. 실험 계획

### 파라미터 분석 결과 (New tanh-based formula)

실제 HDMAP entropy 범위: **0.40 ~ 0.47** (매우 좁음)
- Normal 평균: 0.427
- Abnormal 평균: 0.450
- 차이: 0.023

**새 수식 특성**:
- `sensitivity=0` → `dropout = 0.3` (base_dropout, 기존 Dinomaly와 동일)
- `normal_entropy=0.43` 기준으로 dropout이 조절됨
- Normal 샘플: entropy < normal_entropy → dropout 증가
- Abnormal 샘플: entropy > normal_entropy → dropout 감소

| Sensitivity | Normal Dropout | Abnormal Dropout | 차이 |
|-------------|----------------|------------------|------|
| 0 | 0.300 | 0.300 | 0.000 |
| 4 | 0.304 | 0.276 | 0.028 |
| 15 (권장) | 0.313 | 0.242 | 0.072 |
| 20 | 0.316 | 0.229 | 0.087 |

### 조건
- **Dataset**: HDMAP PNG only (FFT는 orientation entropy가 부적합)
- **Domain**: domain_A
- **Steps**: 200, 1000

### 실험 케이스 (수정됨)

| Case | Model | sensitivity | min/max dropout | 비고 |
|------|-------|-------------|-----------------|------|
| baseline | dinomaly | - | 0.2 (fixed) | 기존 결과 있음 |
| adaptive_mid | dinomaly_adaptive | 15.0 | 0.1 / 0.6 | 중간 강도 |
| adaptive_high | dinomaly_adaptive | 20.0 | 0.1 / 0.6 | 높은 강도 |

---

## 6. 실험 명령어

### 로그 디렉토리 생성
```bash
mkdir -p logs
```

### 명령줄 옵션
```
--model              : dinomaly | dinomaly_adaptive
--domain             : domain_A | domain_B | domain_C | domain_D | all
--max-steps          : 학습 스텝 수
--gpu                : GPU ID
--output-dir         : 결과 저장 경로
--base-dropout       : 기준 dropout (default: 0.3)
--min-dropout        : 최소 dropout (default: 0.1)
--max-dropout        : 최대 dropout (default: 0.6)
--sensitivity        : entropy→dropout 민감도 (default: 4.0, 0이면 base_dropout 고정)
--normal-entropy     : 정상 샘플 기준 entropy (default: 0.43)
```

### Case 1: Adaptive Mid (sensitivity=15.0, 권장)

```bash
# domain_A, 200 steps (GPU 0)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive \
    --domain domain_A \
    --max-steps 200 \
    --gpu 0 \
    --sensitivity 15.0 \
    --output-dir results/hdmap_adaptive_validation/sens15 \
    > logs/adaptive_sens15_domainA_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# domain_A, 1000 steps (GPU 1)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive \
    --domain domain_A \
    --max-steps 1000 \
    --gpu 1 \
    --sensitivity 15.0 \
    --output-dir results/hdmap_adaptive_validation/sens15 \
    > logs/adaptive_sens15_domainA_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Case 2: Adaptive High (sensitivity=20.0)

```bash
# domain_A, 200 steps (GPU 2)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive \
    --domain domain_A \
    --max-steps 200 \
    --gpu 2 \
    --sensitivity 20.0 \
    --output-dir results/hdmap_adaptive_validation/sens20 \
    > logs/adaptive_sens20_domainA_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# domain_A, 1000 steps (GPU 3)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive \
    --domain domain_A \
    --max-steps 1000 \
    --gpu 3 \
    --sensitivity 20.0 \
    --output-dir results/hdmap_adaptive_validation/sens20 \
    > logs/adaptive_sens20_domainA_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Case 3: Ablation - sensitivity=0 (base_dropout 고정, 기존 Dinomaly와 동일)

```bash
# sensitivity=0이면 dropout이 항상 base_dropout(0.3)으로 고정됨
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive \
    --domain domain_A \
    --max-steps 1000 \
    --gpu 0 \
    --sensitivity 0 \
    --output-dir results/hdmap_adaptive_validation/sens0_ablation \
    > logs/adaptive_sens0_domainA_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 7. 결과 확인

### 로그 확인
```bash
# 실행 중인 프로세스
ps aux | grep hdmap_adaptive | grep -v grep

# 최신 로그 tail
tail -f logs/$(ls -t logs/ | head -1)
```

### 결과 파일
```bash
# 결과 JSON 확인
cat results/hdmap_adaptive_validation/*/domain_A_*/results.json | python -m json.tool
```

---

## 8. 예상 결과

### 가설
1. **200 steps**: Adaptive와 Standard 성능 차이 적음 (아직 과적합 전)
2. **1000+ steps**: Adaptive가 더 높은 AUROC 유지 (과적합 방지 효과)
3. **Aggressive 설정**: 더 강한 regularization, 학습 초기 성능은 낮을 수 있음

### 성공 기준
- 1000 steps에서 Adaptive AUROC > Standard AUROC
- Adaptive 모델의 step 증가에 따른 성능 하락 완화

---

## 9. 향후 계획

1. **실험 완료 후**: 결과 분석 및 hyperparameter 튜닝
2. **효과 확인 시**: 다른 도메인 (B, C, D)으로 확장
3. **장기 학습 테스트**: 5000, 10000 steps에서 효과 검증
