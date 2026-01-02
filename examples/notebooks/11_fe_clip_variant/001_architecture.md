# FE-CLIP 아키텍처 이해하기

> CLIP과 WinCLIP을 알고 있는 사람을 위한 FE-CLIP 설명서

## 1. 핵심 아이디어: 왜 FE-CLIP인가?

### WinCLIP의 한계

WinCLIP은 CLIP의 frozen 특징을 그대로 사용:
```
Image → CLIP ViT (frozen) → patch tokens → text와 비교 → anomaly score
```

**문제점**: CLIP은 semantic 정보에 최적화됨 (개 vs 고양이).
anomaly detection에 필요한 **local texture/frequency 정보**가 부족.

### FE-CLIP의 해결책

CLIP 특징에 **주파수 정보를 주입**하는 learnable adapter 추가:
```
Image → CLIP ViT blocks → [FFE + LFS adapters] → enhanced features → anomaly score
```

**핵심**: CLIP backbone은 frozen, adapter만 학습 → 적은 파라미터로 효과적 개선

---

## 2. 전체 아키텍처 흐름

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FE-CLIP Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Image (336×336)                                                     │
│       │                                                               │
│       ▼                                                               │
│  ┌─────────────────┐                                                 │
│  │ Patch Embedding │  conv1: 14×14 patches → 576 tokens             │
│  │   + Pos Embed   │  각 토큰: 1024-dim (ViT-L)                      │
│  └────────┬────────┘                                                 │
│           │                                                           │
│           ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              ViT Transformer (24 blocks, ViT-L)               │    │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐      │    │
│  │  │ Block 0 │→→→│ Block 1 │→→→│  ...    │→→→│Block 23 │      │    │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘      │    │
│  │       │             │             │             │            │    │
│  │       ▼             ▼             ▼             ▼            │    │
│  │  [N=4개 tap 위치 선택하여 adapter 적용 (예: [15,18,21,23])]  │    │
│  │                                                               │    │
│  │       ┌─────────────────────────────────────┐                │    │
│  │       │         Adapter Block               │                │    │
│  │       │  ┌─────┐     ┌─────┐               │                │    │
│  │       │  │ FFE │  +  │ LFS │  → fused      │                │    │
│  │       │  └─────┘     └─────┘               │                │    │
│  │       └─────────────────────────────────────┘                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│           │                                                           │
│           ▼                                                           │
│  ┌─────────────────────────────────────────┐                        │
│  │  Output per tap block:                   │                        │
│  │  • CLS token → fc_clip → image score    │                        │
│  │  • Patch tokens → fc_patch → pixel map  │                        │
│  └─────────────────────────────────────────┘                        │
│           │                                                           │
│           ▼                                                           │
│  Average across tap blocks → Final score & map                       │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

**소스 코드 참조**: `torch_model.py:238-324` (`forward_tokens` 메서드)

---

## 3. 핵심 컴포넌트 상세

### 3.1 FFE Adapter (Frequency-aware Feature Extraction)

**목적**: Non-overlapping window에서 DCT로 주파수 정보 추출

```python
# adapters.py:65-126
class FFEAdapter(nn.Module):
    def __init__(self, d_model: int, P: int = 3):
        self.P = P  # window size (default: 3×3)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
```

**동작 과정**:
```
Input: patch tokens (B, H, W, D)  # H=W=24 for 336px input
    │
    ▼
1. Reshape to (P×P) windows: (B, H/P, W/P, P, P, D)
    │
    ▼
2. 2D DCT 적용 (각 window별 주파수 분해)
    │
    ▼
3. MLP 처리 (주파수 도메인에서 학습)
    │
    ▼
4. 2D IDCT로 공간 도메인 복원
    │
    ▼
Output: (B, H, W, D)
```

**직관적 이해**:
- DCT는 JPEG 압축에 사용되는 변환
- 저주파 = 전체적 밝기/색상, 고주파 = edge/texture
- MLP가 어떤 주파수 성분이 anomaly에 중요한지 학습

### 3.2 LFS Adapter (Local Frequency Statistics)

**목적**: Sliding window로 local 주파수 통계 추출

```python
# adapters.py:128-215
class LFSAdapter(nn.Module):
    def __init__(self, d_model: int, Q: int = 3):
        self.Q = Q  # window size (default: 3×3)
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
```

**동작 과정**:
```
Input: patch tokens (B, H, W, D)
    │
    ▼
1. Sliding window로 (Q×Q) 패치 추출 (with padding)
    │
    ▼
2. 각 위치에서 2D DCT 적용
    │
    ▼
3. DCT 계수들의 평균 계산 (주파수 통계)
    │
    ▼
4. Conv + GELU로 공간 관계 학습
    │
    ▼
Output: (B, H, W, D)
```

**FFE vs LFS 차이점**:
| 특성 | FFE | LFS |
|------|-----|-----|
| Window 방식 | Non-overlapping | Sliding (overlapping) |
| 출력 의미 | 변환된 특징 | 주파수 통계량 |
| 공간 해상도 | 유지 | 유지 |

### 3.3 Feature Fusion

```python
# torch_model.py:294-295
patch_hat = self.lambda_fuse * (ffe_out + lfs_out) + (1 - self.lambda_fuse) * patch
```

- `lambda_fuse = 0.1` (default)
- 원본 CLIP 특징의 90% + adapter 특징의 10%
- **보수적 융합**: CLIP의 semantic 정보를 최대한 보존

---

## 4. WinCLIP과의 결정적 차이

### 4.1 특징 추출 방식

| 구분 | WinCLIP | FE-CLIP |
|------|---------|---------|
| Backbone | Frozen CLIP | Frozen CLIP |
| 추가 모듈 | 없음 | FFE + LFS adapters |
| 학습 파라미터 | 없음 (zero-shot) | Adapter만 학습 |
| 주파수 정보 | 없음 | DCT로 명시적 추출 |

### 4.2 Anomaly Score 계산

**WinCLIP**:
```python
# 마지막 block의 특징만 사용
features = clip_visual(image)[-1]
score = cosine_sim(features, text_embeddings)
```

**FE-CLIP**:
```python
# torch_model.py:343-347
# 여러 tap block에서 특징 추출 후 평균
scores, maps = self.forward_tokens(images)
score = torch.stack(scores, dim=0).mean(dim=0)  # N개 block 평균
amap = torch.stack(maps, dim=0).mean(dim=0)
```

### 4.3 학습 방식

**WinCLIP**: 학습 없음 (순수 zero-shot)

**FE-CLIP**: Cross-dataset 학습
```python
# 예: VisA로 학습 → MVTec에서 평가
# losses.py:80-122
L_total = w_cls * L_cls + w_mask * (L_focal + L_dice)
```
- `L_cls`: Image-level BCE loss
- `L_focal + L_dice`: Pixel-level segmentation loss

---

## 5. Tap Block 선택의 중요성

### 5.1 왜 여러 block에서 tap하는가?

> **용어 정리**: ViT-L은 24개의 transformer block을 가짐.
> 논문은 이 중 N=4개의 block을 선택하여 adapter를 적용/평균한다고 설명.
> (N은 "tap할 block 개수"이지, ViT 전체 block 수가 아님)

ViT의 각 block은 다른 수준의 정보를 인코딩:
- **초반 block (0-8)**: low-level 특징 (edge, texture)
- **중반 block (9-16)**: mid-level 특징 (parts, patterns)
- **후반 block (17-23)**: high-level 특징 (semantic)

**Anomaly detection에는 다양한 레벨이 필요**:
- Texture anomaly → 초반 block 중요
- Structural anomaly → 중반 block 중요
- Object-level anomaly → 후반 block 중요

### 5.2 우리 실험 결과 (Exp14-16)

> **논문 참고**: 논문은 "N=4개의 블록 레벨에서 특징을 추출/평균한다"고만 밝히며,
> 구체적인 tap 위치(어떤 블록 인덱스)는 명시하지 않음.

| Config | Tap Indices | AUROC | pAUROC | 특징 |
|--------|-------------|-------|--------|------|
| last4 | [20,21,22,23] | 87.8% | 92.6% | 실험 baseline |
| **spread_3** | **[15,18,21,23]** | **88.2%** | **93.7%** | **추천** |
| spread_5 | [8,13,18,23] | 84.4% | 94.1% | pAUROC 최고 |

**발견**:
- 연속 block보다 **spread (비연속)가 더 효과적**
- 다양한 abstraction level 활용이 유리
- `spread_3 [15,18,21,23]`이 균형 최적

**소스 코드 참조**: `torch_model.py:106-116` (tap block 결정 로직)

---

## 6. 학습 파이프라인

### 6.1 학습 대상

```python
# torch_model.py:101-149
# Frozen (학습 안함):
self.clip.eval()
for p in self.clip.parameters():
    p.requires_grad = False

# Trainable (학습함):
self.ffe = nn.ModuleList([FFEAdapter(width, P=P) for _ in self.tap_blocks])
self.lfs = nn.ModuleList([LFSAdapter(width, Q=Q) for _ in self.tap_blocks])
self.fc_patch = nn.Linear(width, text_dim)
```

> **논문 모호함 주의**:
> - Section 3.2: "single **learnable** fc를 사용해 patch를 text 공간에 align"
> - Section 3.5: "**Only** FFE/LFS adapters만 optimized"
>
> 이 두 서술이 상충되어 fc_patch 학습 여부가 모호함.
> 우리 구현에서는 fc_patch를 학습하되, lr을 크게 낮추는 것이 효과적이었음.

> **구현 주의**: `train()` 메서드를 오버라이드하여 CLIP을 항상 eval 모드로 유지함.
> 이유: CLIP은 frozen이지만, train 모드에서 dropout이 활성화되어 adapter가
> "noisy teacher"에서 학습하게 되어 성능이 저하됨. (`torch_model.py:158-176`)

### 6.2 최적 학습 설정 (실험 기반)

```python
# 001_result.md에서 정리
epochs = 9
batch_size = 16
adapter_lr = 5e-4
fc_patch_lr = 5e-6  # adapter_lr / 100 (Exp12에서 발견, 우리 구현 선택)
```

**fc_patch lr을 낮추는 이유** (우리 실험 결과):
- fc_patch는 patch token → text space 매핑
- 너무 빨리 학습하면 CLIP의 text alignment가 깨짐
- 1/100로 낮추면 pAUROC +1.3% 개선

---

## 7. 추론 파이프라인

```python
# torch_model.py:326-357
@torch.no_grad()
def forward(self, images):
    scores, maps = self.forward_tokens(images)

    # 1. 여러 tap block의 score 평균
    score = torch.stack(scores, dim=0).mean(dim=0)  # (B,)

    # 2. 여러 tap block의 map 평균
    amap = torch.stack(maps, dim=0).mean(dim=0)  # (B, Ht, Wt)

    # 3. Upsampling to input resolution
    amap = F.interpolate(amap, size=images.shape[-2:], mode="bilinear")

    return InferenceBatch(pred_score=score, anomaly_map=amap)
```

### Score 계산 상세

```python
# torch_model.py:219-236
def prob_abnormal(self, z):
    z = F.normalize(z, dim=-1)
    logits = (z @ self.text_emb.t()) / self.temperature  # τ=0.07
    return logits.softmax(dim=-1)[..., 1]  # abnormal class 확률
```

- `text_emb`: ["normal", "abnormal"] 프롬프트의 CLIP embedding
- Softmax로 abnormal 확률 계산
- Temperature τ=0.07 (논문은 τ를 정의하지만 구체적 값은 미명시, CLIP 관행값 사용)

---

## 8. 요약: CLIP → WinCLIP → FE-CLIP 진화

```
CLIP (zero-shot classification)
  │
  │  "patch token도 text와 비교하면 pixel-level anomaly 가능"
  ▼
WinCLIP (zero-shot anomaly detection)
  │
  │  "CLIP 특징에 주파수 정보를 더하면 더 정확"
  ▼
FE-CLIP (frequency-enhanced anomaly detection)  [논문]
  │
  │  • FFE: Non-overlapping DCT로 global frequency
  │  • LFS: Sliding DCT로 local frequency statistics
  │  • Cross-dataset 학습으로 일반화 향상
  │  • N=4 tap blocks (위치 미명시)
  ▼
우리 구현 (anomalib)  [실험적 개선]
  │
  │  • tap block 위치 탐색 → spread_3 [15,18,21,23] 추천
  │  • fc_patch lr 정책 → 1/100 낮춤 (논문 모호, 실험으로 결정)
  │  • train() 오버라이드로 CLIP dropout 비활성화
  │  • Macro-average 평가 (논문 방식 확인)
  ▼
MVTec: 90.7% AUROC, 91.1% pAUROC
VisA:  88.2% AUROC, 93.7% pAUROC
```

---

## 참조 파일

| 파일 | 설명 |
|------|------|
| `torch_model.py` | FEClipModel 핵심 구현 |
| `adapters.py` | FFE, LFS adapter 구현 |
| `losses.py` | BCE, Focal, Dice loss |
| `prompting.py` | Text prompt 생성 |
| `001_result.md` | 실험 결과 및 최적 설정 |
