# Directional Attention for DINOMALY Improvement

## 배경 및 동기

### 현재 상황
- **DINOMALY 성능**: 0.911 AUROC (현재 최고 성능)
- **DRAEM SevNet 한계**: 0.7~0.8 AUROC 정도의 제한적 성능
- **데이터 특성**: HDMap 기어박스 진동 데이터, 고장 시 가로선 패턴 발생

### 핵심 아이디어
기존 DINOMALY의 Linear Attention을 물리적 고장 특성에 맞는 **Directional Attention**으로 개선

## 기술적 상세

### 1. 현재 Linear Attention 분석
```python
# 기존 DINOMALY Linear Attention 수식
Q, K, V = qkv_projection(X)
Q' = ELU(Q) + 1.0  # 양수 특성 보장
K' = ELU(K) + 1.0

# Linear attention 핵심 수식
KV = K'^T @ V                    # (d_head, d_head) - 전역 상호작용
K_sum = sum(K', dim=seq)         # 정규화 항
Z = 1 / (Q' @ K_sum)            # 정규화
Output = (Q' @ KV) * Z          # 최종 출력
```

### 2. Directional Linear Attention 구현안
```python
class DirectionalLinearAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8, spatial_shape=(16,16)):
        super().__init__()
        self.H, self.W = spatial_shape
        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self.alpha_learnable = nn.Parameter(torch.tensor(0.5))  # 학습 가능한 가중치
        
    def forward(self, x):
        # x: (B, H*W, D)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D//self.num_heads)
        q, k, v = qkv.unbind(2)  # 각각 (B, N, heads, d_head)
        
        q, k = F.elu(q) + 1.0, F.elu(k) + 1.0
        
        # Spatial reshape: (B, heads, H, W, d_head)
        q = q.transpose(1,2).reshape(B, self.num_heads, self.H, self.W, -1)
        k = k.transpose(1,2).reshape(B, self.num_heads, self.H, self.W, -1)
        v = v.transpose(1,2).reshape(B, self.num_heads, self.H, self.W, -1)
        
        # Y-axis attention (가로선 특화)
        kv_y = torch.matmul(k.transpose(-3,-2), v)  # (B, heads, W, d_head, d_head)
        k_sum_y = k.sum(dim=-3, keepdim=True)       # (B, heads, 1, W, d_head)
        z_y = 1.0 / torch.sum(q * k_sum_y, dim=-1, keepdim=True)
        out_y = torch.matmul(q, kv_y) * z_y
        
        # X-axis attention (세로선 특화) 
        kv_x = torch.matmul(k.transpose(-4,-2), v)  # (B, heads, H, d_head, d_head)
        k_sum_x = k.sum(dim=-2, keepdim=True)       # (B, heads, H, 1, d_head)
        z_x = 1.0 / torch.sum(q * k_sum_x, dim=-1, keepdim=True)
        out_x = torch.matmul(q, kv_x) * z_x
        
        # Adaptive combination
        alpha = torch.sigmoid(self.alpha_learnable)  # 0~1 범위 보장
        output = alpha * out_y + (1-alpha) * out_x
        
        return output.reshape(B, N, D)
```

### 3. 적용 전략
- **부분 적용**: 첫 2-3개 decoder layer에만 적용 (spatial structure 유지)
- **나머지 layer**: 기존 LinearAttention 유지
- **Ablation Study**: X-only, Y-only, Combined 비교

## 물리적 근거

### HDMap 데이터 특성
- **Driving axis (Y축)**: 기어박스 driving gear 위치
- **Driven axis (X축)**: driven gear 위치  
- **고장 패턴**: 
  - Driving gear 고장 → Y축 고정, X축 전체에 진동 증가 → **가로선**
  - Driven gear 고장 → X축 고정, Y축 전체에 진동 증가 → **세로선**

### 기존 Linear Attention 한계
- 전역적 attention으로 방향성 무시
- 물리적 고장 특성과 architecture 불일치

## 논문 기여도 분석

### 강점
- **Novel Architecture Innovation**: 물리적 domain knowledge → attention mechanism 직접 반영
- **Strong Theoretical Foundation**: 명확한 물리적 근거
- **Generalizable**: 다른 structured pattern anomaly에도 적용 가능
- **Clear Story**: "Physical Failure Patterns require Directional Attention"

### 예상 리뷰어 질문 및 대답
**Q**: *"Why not use CNN-based directional convolutions?"*  
**A**: Transformer의 long-range dependency + directional bias 결합이 핵심

**Q**: *"How does this compare to positional encoding approaches?"*  
**A**: 실험으로 비교 (positional encoding은 위치만, 우리는 방향성)

## 구현 복잡도
- **난이도**: 중간 (기존 LinearAttention 수정 + spatial reshape)
- **구현 시간**: 약 2주
- **주요 챌린지**: Spatial dimension handling, gradient flow 최적화

## 실험 계획

### Ablation Studies
1. Vanilla Linear Attention vs Directional
2. X-only vs Y-only vs X+Y combined
3. Different α strategies: 학습 vs 고정 vs adaptive
4. Layer-wise application: 어떤 layer에 적용이 최적인가

### Generalization Test
- 다른 industrial dataset에서 검증
- Medical imaging, Satellite imagery 등 확장 가능성

## 다음 단계 TODO
1. **[Week 1-2]** Basic DirectionalLinearAttention 모듈 구현
2. **[Week 2-3]** DINOMALY에 통합 및 첫 실험
3. **[Week 3-4]** Ablation study 및 hyperparameter tuning
4. **[Week 4-5]** 다른 dataset 일반화 실험
5. **[Week 6-8]** 논문 작성 및 submission 준비

## 타겟 학회
- **1순위**: ICLR 2026
- **2순위**: NeurIPS 2025
- **3순위**: AAAI 2026

## 예상 논문 제목
- "Directional Attention for Structure-Aware Anomaly Detection"
- "Physical-Informed Attention Mechanisms in Vision Transformers for Industrial Anomaly Detection"