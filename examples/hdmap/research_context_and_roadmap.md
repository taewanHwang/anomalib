# HDMap Anomaly Detection Research Context & Roadmap

## 전체 컨텍스트

### 연구 배경
- **도메인**: HDMap 기어박스 진동 데이터 이상감지
- **데이터 특성**: 
  - 정상 → 고장 시 **additive pattern** 추가 (주로 가로선)
  - Driving axis (Y축) vs Driven tooth (X축) 조합
  - 물리적 고장: 특정 축에서 전체 진동 증가 → 방향성 패턴

### 현재 성능 현황
```
📊 실험 결과 요약 (AUROC):
- DINOMALY: 0.911 (최고 성능) ✅
- DRAEM SevNet: 0.6~0.8 (제한적 성능) ❌
- PatchCore: 0.8 정도
```

### 연구 목표
**DINOMALY를 기반으로 HDMap 특화 개선을 통해 세계적 수준의 논문 작성**

## 핵심 아이디어 요약

### 🎯 Main Contribution: Directional Attention
**Physical-Informed Architecture로 가장 유망**

**배경**: 기존 Linear Attention은 전역적이라 방향성 고장 패턴 무시  
**해법**: X축/Y축 별도 attention + 학습 가능한 조합

```python
# 핵심 수식
out_y = DirectionalAttention_Y(q, k, v)  # 가로선 특화
out_x = DirectionalAttention_X(q, k, v)  # 세로선 특화  
output = α * out_y + (1-α) * out_x      # 학습 가능한 조합
```

**논문 기여도**: ★★★★★
- Novel architecture innovation
- Strong physical foundation
- Generalizable to other domains

### 🎯 Secondary: Severity-Aware DINOMALY
**실용적 가치는 높으나 기술적 독창성 제한**

**배경**: 산업 현장에서 anomaly 심각도 정보 필요  
**해법**: Multi-task learning (detection + severity estimation)

**논문 기여도**: ★★★
- Practical value 높음
- 기술적 novelty 제한적

## 상세 분석 결과

### DINOMALY 아키텍처 분석
```
📁 /anomalib/src/anomalib/models/image/dinomaly/
├── README.md              # 아키텍처 개요
├── lightning_model.py     # PyTorch Lightning wrapper
├── torch_model.py         # 핵심 모델 구현  
└── components/
    ├── layers.py          # LinearAttention 구현 ⭐
    ├── loss.py           # CosineHardMiningLoss
    └── vision_transformer.py
```

**핵심 특징**:
1. **DINOv2 Encoder** (frozen) + **Bottleneck MLP** + **ViT Decoder**
2. **Linear Attention**: ELU 기반 positive feature maps
3. **Hard Mining Loss**: Easy sample down-weighting
4. **Multi-scale Feature Fusion**: Layer group별 reconstruction

### Linear Attention 수식 분석
```python
# 현재 구현 (layers.py:107-146)
Q' = ELU(Q) + 1.0
K' = ELU(K) + 1.0
KV = K'^T @ V                    # Global interaction
K_sum = sum(K', dim=seq)         # Normalization
Z = 1 / (Q' @ K_sum)
Output = (Q' @ KV) * Z
```

## Implementation Roadmap

### Phase 1: Directional Attention (4주)
```
Week 1-2: Core Implementation
- [ ] DirectionalLinearAttention 모듈 구현
- [ ] DINOMALY integration (decoder layers 2-3개만 교체)
- [ ] Basic training pipeline 구축

Week 3-4: Experimental Validation  
- [ ] Ablation studies (X-only, Y-only, Combined)
- [ ] α combination strategy 최적화
- [ ] HDMap 데이터에서 첫 성능 검증
```

### Phase 2: Advanced Experiments (4주)
```
Week 5-6: Deep Analysis
- [ ] Layer-wise application 최적화 
- [ ] Attention visualization 및 해석
- [ ] 다양한 spatial_shape 실험

Week 7-8: Generalization
- [ ] 다른 industrial dataset 검증
- [ ] Medical/Satellite imagery 확장 실험
- [ ] Comparative analysis with baselines
```

### Phase 3: Paper Writing (4주)
```
Week 9-10: Core Writing
- [ ] Introduction & Related Work
- [ ] Method & Architecture 상세 기술
- [ ] Experimental setup 및 results

Week 11-12: Polish & Submit
- [ ] Discussion & Limitation
- [ ] Revision 및 proofreading  
- [ ] Target venue submission (ICLR/NeurIPS)
```

## 파일 구조 계획

### 구현 파일 구조
```
📁 anomalib/src/anomalib/models/image/dinomaly_directional/
├── __init__.py
├── lightning_model.py          # DirectionalDINOMALY Lightning model
├── torch_model.py             # Core model with DirectionalAttention
└── components/
    ├── __init__.py
    ├── directional_attention.py   # 🆕 DirectionalLinearAttention
    ├── layers.py                  # Extended from original
    └── loss.py                    # 기존 loss 재사용
```

### 실험 설정 파일
```
📁 examples/hdmap/configs/
├── dinomaly_directional_base.yaml
├── dinomaly_directional_ablation.yaml
└── dinomaly_severity_aware.yaml
```

## 핵심 실험 설계

### Ablation Studies Plan
1. **Direction Comparison**
   - Vanilla LinearAttention (baseline)
   - X-axis only attention  
   - Y-axis only attention
   - Combined (α learnable vs fixed)

2. **Architecture Variants**
   - 어떤 decoder layer에 적용? (1-2층 vs 2-3층 vs 전체)
   - Spatial resolution 영향? (16x16 vs 32x32)
   - α initialization strategy

3. **Generalization Tests**
   - Cross-domain: 다른 기계 데이터
   - Cross-resolution: 다른 이미지 크기
   - Robustness: Noise, rotation 등

### Evaluation Metrics
- **Primary**: AUROC, AUPR (기존 anomaly detection metric)
- **Secondary**: F1-score, Precision@90%Recall
- **Analysis**: Attention map visualization, inference time

## 논문 전략

### Target Venues (우선순위)
1. **ICLR 2026** - Architecture innovation 인정
2. **NeurIPS 2025** - ML conference 최고 권위
3. **AAAI 2026** - Practical application 중시

### Paper Title 후보
- "Directional Attention for Structure-Aware Anomaly Detection"
- "Physical-Informed Attention Mechanisms in Vision Transformers"  
- "Structure-Aware Linear Attention for Industrial Anomaly Detection"

### Contribution Claims
1. **Novel Directional Attention**: 물리적 패턴에 특화된 attention mechanism
2. **Physical Foundation**: Domain knowledge → Architecture design 직접 연결
3. **Broad Applicability**: Industrial/Medical imaging 전반 적용 가능
4. **Strong Empirical Results**: HDMap + 다른 도메인에서 consistent improvement

## 위험 관리

### Technical Risks
- **Gradient Flow**: Spatial reshape 과정에서 gradient vanishing/exploding
- **Memory Usage**: Directional attention의 computational overhead  
- **Generalization**: HDMap 특화가 다른 도메인에서 성능 저하 가능

### Mitigation Strategies
- **Progressive Implementation**: 단계별 구현으로 문제 조기 발견
- **Extensive Ablation**: 각 component별 영향 분석
- **Multiple Baselines**: 충분한 비교군 확보

## 다음 세션 시작 시 참고사항

### 중요 결정사항
- ✅ **Main approach**: Directional Attention 선택
- ✅ **Target**: ICLR 2026 submission  
- ✅ **Implementation priority**: DirectionalLinearAttention 먼저

### 핵심 파일 위치
- `/examples/hdmap/directional_attention_improvement.md` - 상세 기술 설계
- `/examples/hdmap/severity_aware_dinomaly.md` - 대안 접근법
- `/examples/hdmap/research_context_and_roadmap.md` - 전체 컨텍스트 (현재 파일)

### 바로 시작할 수 있는 작업
1. **DirectionalLinearAttention 모듈 구현**
2. **기존 DINOMALY 코드 분석 및 integration point 파악**  
3. **First prototype training on HDMap data**

---

**🚀 Ready to Start Implementation!**