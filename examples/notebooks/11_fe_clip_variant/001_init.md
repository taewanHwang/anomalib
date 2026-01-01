아래 문서는 **anomalib의 WinCLIP 구현 스타일**을 최대한 유지하면서, **FE-CLIP을 새 모델로 추가**하고, **MVTec AD로 adapter(FFE/LFS + 일부 projection)만 파인튜닝한 뒤 현장 데이터에 적용**하는 전체 가이드입니다.
(코드는 “바로 붙여서 시작할 수 있는 수준”으로 적되, `open_clip`의 ViT 내부 속성명은 버전에 따라 다를 수 있으니 **확인 포인트**를 같이 넣었습니다.)

---

# 1. 목표와 구현 전략

## 목표

* **CLIP 본체(visual/text encoder)는 freeze**
* **FFE/LFS adapter + (필요 시) patch→text projection(fc_patch)만 학습**
* MVTec AD의 (라벨/마스크가 있는) 데이터로 adapter를 학습 → **현장 데이터에 zero-shot 적용**

## 왜 anomalib WinCLIP을 참고하나?

* anomalib에서 이미 `open_clip` 로딩, 전처리, Lightning/Engine 흐름이 갖춰져 있음
* 모델 추가 시 `lightning_model.py` / `torch_model.py` 분리 패턴이 잘 잡혀 있음

## WinCLIP 대비 FE-CLIP에서 가장 큰 차이

* WinCLIP: sliding window + harmonic aggregation 중심(“윈도우 임베딩”)
* FE-CLIP: **비전 인코더의 중간 블록 토큰(feature)을 adapter로 바꾸는 게 핵심**
  → `open_clip` ViT 내부 block loop를 직접 돌면서 tokens를 수정하는 “커스텀 forward”가 필요

---

# 2. 폴더/파일 구조 (WinCLIP과 동일 패턴)

`src/anomalib/models/image/feclip/`

```
feclip/
├── README.md
├── __init__.py
├── lightning_model.py
├── torch_model.py
├── adapters.py
├── prompting.py
└── losses.py
```

* `prompting.py`: 정상/고장 프롬프트
* `adapters.py`: FFE/LFS + (필요시) masked LFS
* `torch_model.py`: CLIP 로드/Freeze + 커스텀 ViT forward + score/map 생성
* `lightning_model.py`: 학습/검증 step + optimizer
* `losses.py`: BCE, focal, dice

---

# 3. prompting.py (논문 방식: 2개 프롬프트로 시작)

```python
# prompting.py
def feclip_prompts() -> list[str]:
    # normal=0, abnormal=1
    return ["A photo of a normal object", "A photo of a damaged object"]
```

> 현장 적용 시 “object”를 도메인 명사로 바꿔도 되지만(예: “a normal PCB”),
> 일단 논문 방식 그대로 2개로 시작하는 게 구현/디버깅에 좋습니다.

---

# 4. adapters.py (FFE / LFS 핵심 구현)

## 4.1 DCT를 어떻게 구현할까?

PyTorch는 기본 DCT가 표준화되어 있지 않아서, **작은 P,Q(3~5)**에서는 **DCT basis 행렬 곱** 방식이 간단하고 충분히 빠릅니다.

## 4.2 FFE: non-overlap P×P token window에 DCT → Linear+GELU → iDCT

중요 포인트:

* 토큰 그리드(H×W)가 P로 나누어 떨어지지 않을 수 있음(예: 14×14에서 P=3)
  → **token-grid padding 후 다시 crop** 로 처리하는 게 안전

## 4.3 LFS: Q×Q sliding window로 SW-DCT → Q×Q “그룹 평균(mean)” → Conv+GELU

중요 포인트:

* LFS는 구조상 “이웃을 모으는 방식”이 논문에서 **mean**으로 명시되어 있음
* 현장 데이터에 letterbox padding이 크면, LFS 집계에 padding 토큰이 섞여 **희석**될 수 있음
  → (옵션) **valid_mask 기반 masked mean LFS**를 권장

아래 코드는 바로 사용 가능한 뼈대입니다.

```python
# adapters.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def dct_basis(n: int, device, dtype) -> torch.Tensor:
    k = torch.arange(n, device=device, dtype=dtype).view(-1, 1)
    i = torch.arange(n, device=device, dtype=dtype).view(1, -1)
    C = torch.cos(math.pi / n * (i + 0.5) * k)
    C[0] *= 1.0 / math.sqrt(n)
    C[1:] *= math.sqrt(2.0 / n)
    return C  # (n,n)

def dct2(x: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    # x: (..., n, n)  ->  C x C^T
    y = torch.einsum("ab,...bc->...ac", C, x)
    y = torch.einsum("...ab,cb->...ac", y, C)
    return y

def idct2(x: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    # orthonormal inverse: C^T x C
    y = torch.einsum("ba,...bc->...ac", C, x)
    y = torch.einsum("...ab,bc->...ac", y, C)
    return y

class FFEAdapter(nn.Module):
    def __init__(self, d_model: int, P: int = 3):
        super().__init__()
        self.P = P
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # f: (B,H,W,D)
        B, H, W, D = f.shape
        P = self.P

        # pad token-grid so H,W divisible by P
        Hp = (H + P - 1) // P * P
        Wp = (W + P - 1) // P * P
        if Hp != H or Wp != W:
            pad_h = Hp - H
            pad_w = Wp - W
            f = F.pad(f, (0, 0, 0, pad_w, 0, pad_h), mode="replicate")  # pad W then H

        B, H2, W2, D = f.shape
        C = dct_basis(P, device=f.device, dtype=f.dtype)

        # (B, H/P, W/P, P, P, D)
        x = f.view(B, H2 // P, P, W2 // P, P, D).permute(0, 1, 3, 2, 4, 5)
        # -> (B,Hp,Wp,D,P,P)
        x = x.permute(0, 1, 2, 5, 3, 4)

        x = dct2(x, C)                              # (B,Hp,Wp,D,P,P)
        x = x.permute(0, 1, 2, 4, 5, 3)             # (B,Hp,Wp,P,P,D)
        x = self.mlp(x)
        x = x.permute(0, 1, 2, 5, 3, 4)             # (B,Hp,Wp,D,P,P)
        x = idct2(x, C)                             # (B,Hp,Wp,D,P,P)
        x = x.permute(0, 1, 2, 4, 5, 3)             # (B,Hp,Wp,P,P,D)

        out = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H2, W2, D)

        # crop back if padded
        return out[:, :H, :W, :]

class LFSAdapter(nn.Module):
    def __init__(self, d_model: int, Q: int = 3, conv_kernel: int = 3, pad_mode: str = "replicate"):
        super().__init__()
        self.Q = Q
        self.pad_mode = pad_mode
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.GELU(),
        )

    def forward(self, f: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        f: (B,H,W,D)
        valid_mask(optional): (B,H,W) bool or {0,1} float
          - True/1 = real image region (not letterbox padding)
          - used for masked mean over neighbors (추천)
        """
        B, H, W, D = f.shape
        Q = self.Q
        C = dct_basis(Q, device=f.device, dtype=f.dtype)

        # (B,H,W,D) -> (B,D,H,W)
        x = f.permute(0, 3, 1, 2).contiguous()
        pad = Q // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode=self.pad_mode)
        patches = F.unfold(x_pad, kernel_size=Q, stride=1)  # (B, D*Q*Q, H*W)
        patches = patches.view(B, D, Q, Q, H, W).permute(0, 4, 5, 1, 2, 3)  # (B,H,W,D,Q,Q)

        # SW-DCT
        fd = dct2(patches, C)  # (B,H,W,D,Q,Q)

        # 집계: mean over (Q,Q). (논문) 
        if valid_mask is None:
            stats = fd.mean(dim=(-1, -2))  # (B,H,W,D)
        else:
            # masked mean: 이웃 중 valid만 반영(현장 letterbox 강추)
            vm = valid_mask.to(fd.dtype)
            vm_pad = F.pad(vm, (pad, pad, pad, pad), mode="constant", value=0)
            vm_nb = F.unfold(vm_pad.unsqueeze(1), kernel_size=Q, stride=1)  # (B, Q*Q, H*W)
            vm_nb = vm_nb.view(B, Q, Q, H, W).permute(0, 3, 4, 1, 2)        # (B,H,W,Q,Q)
            denom = vm_nb.sum(dim=(-1, -2)).clamp(min=1.0)                   # (B,H,W)
            stats = (fd * vm_nb.unsqueeze(3)).sum(dim=(-1, -2)) / denom.unsqueeze(-1)  # (B,H,W,D)

        # Conv+GELU
        y = stats.permute(0, 3, 1, 2).contiguous()
        y = self.conv(y).permute(0, 2, 3, 1).contiguous()
        return y
```

---

# 5. torch_model.py (FE-CLIP 핵심: ViT 커스텀 forward + 블록 평균)

## 5.1 무엇을 해야 하나?

* open_clip으로 CLIP 로드
* CLIP 파라미터 freeze
* ViT forward를 “블록 단위 loop”로 돌면서 tap block에서만 adapter 적용
* 각 tap block에서:

  * patch tokens → anomaly map
  * cls token → image score
* tap block 결과 평균

## 5.2 가장 중요한 구현 난이도 포인트

`open_clip` 버전별로 ViT 내부 속성명이 조금 다릅니다.
따라서 아래의 “tokenize/stem” 부분은 너의 `open_clip` 모델 객체를 출력해서 맞춰야 합니다.

* `visual.conv1`, `visual.class_embedding`, `visual.positional_embedding`
* `visual.ln_pre`, `visual.transformer.resblocks`, `visual.ln_post`, `visual.proj`

아래 코드는 “기본 VisualTransformer 계열”을 가정한 형태입니다.

```python
# torch_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from open_clip.tokenizer import tokenize

from anomalib.data import InferenceBatch
from .prompting import feclip_prompts
from .adapters import FFEAdapter, LFSAdapter

class FEClipModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: str,
        n_taps: int = 4,
        lambda_fuse: float = 0.1,
        P: int = 3,
        Q: int = 3,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_fuse = lambda_fuse

        self.clip, _, self._transform = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
        self.visual = self.clip.visual
        self.clip.eval()

        # freeze CLIP
        for p in self.clip.parameters():
            p.requires_grad = False

        # tap blocks
        resblocks = self.visual.transformer.resblocks
        n_blocks = len(resblocks)
        if n_taps >= n_blocks:
            self.tap_blocks = list(range(n_blocks))
        else:
            self.tap_blocks = torch.linspace(0, n_blocks - 1, steps=n_taps).round().long().tolist()

        width = self.visual.transformer.width  # token dim
        self.ffe = nn.ModuleList([FFEAdapter(width, P=P) for _ in self.tap_blocks])
        self.lfs = nn.ModuleList([LFSAdapter(width, Q=Q) for _ in self.tap_blocks])

        # text emb buffer
        self.register_buffer("_text_emb", torch.empty(0))

        # learnable patch projection to text space (선택: 논문은 patch에 learnable fc)
        text_dim = self.clip.text_projection.shape[1] if hasattr(self.clip, "text_projection") else self.visual.proj.shape[1]
        self.fc_patch = nn.Linear(width, text_dim)

        # cls projection은 CLIP frozen proj 사용
        self.fc_clip = self.visual.proj  # frozen

    def setup_text(self):
        device = next(self.parameters()).device
        prompts = feclip_prompts()
        tok = tokenize(prompts).to(device)
        with torch.no_grad():
            t = self.clip.encode_text(tok)
            t = F.normalize(t, dim=-1)
        self._text_emb = t  # (2, text_dim)

    @property
    def text_emb(self):
        if self._text_emb.numel() == 0:
            raise RuntimeError("call setup_text() before forward")
        return self._text_emb

    def prob_abnormal(self, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        logits = (z @ self.text_emb.t()) / self.temperature
        return logits.softmax(dim=-1)[..., 1]

    def forward_tokens(self, images: torch.Tensor, valid_mask_tokens: torch.Tensor | None = None):
        """
        valid_mask_tokens(optional): (B,Ht,Wt) token-grid mask (letterbox 영역 제외)
        return:
          score_list: [ (B,) ... ]
          map_list:   [ (B,Ht,Wt) ... ]
        """
        # ---- stem/tokenize (open_clip ViT 구조에 맞게 조정 필요)
        x = self.visual.conv1(images)              # (B,C',H',W')
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B,HW,D)
        cls = self.visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype)
        x = torch.cat([cls, x], dim=1)             # (B,1+HW,D)

        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        # grid size
        Ht, Wt = self.visual.grid_size  # (H,W)

        score_list, map_list = [], []
        tap_i = 0

        for bi, blk in enumerate(self.visual.transformer.resblocks):
            x = blk(x)

            if bi in self.tap_blocks:
                patch = x[:, 1:, :].reshape(x.shape[0], Ht, Wt, -1)

                ffe_out = self.ffe[tap_i](patch)
                lfs_out = self.lfs[tap_i](patch, valid_mask=valid_mask_tokens)

                patch_hat = self.lambda_fuse * (ffe_out + lfs_out) + (1 - self.lambda_fuse) * patch
                x = torch.cat([x[:, :1, :], patch_hat.reshape(x.shape[0], -1, x.shape[-1])], dim=1)

                # patch map
                patch_txt = self.fc_patch(patch_hat)       # (B,Ht,Wt,text_dim)
                amap = self.prob_abnormal(patch_txt)       # (B,Ht,Wt)

                # cls score
                cls_tok = x[:, 0, :]
                cls_tok = self.visual.ln_post(cls_tok)
                cls_txt = cls_tok @ self.fc_clip
                score = self.prob_abnormal(cls_txt)        # (B,)

                score_list.append(score)
                map_list.append(amap)
                tap_i += 1

        return score_list, map_list

    def forward(self, images: torch.Tensor, valid_mask_tokens: torch.Tensor | None = None):
        scores, maps = self.forward_tokens(images, valid_mask_tokens)
        score = torch.stack(scores, dim=0).mean(dim=0)                 # (B,)
        amap  = torch.stack(maps, dim=0).mean(dim=0)                   # (B,Ht,Wt)
        amap  = F.interpolate(amap.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        return InferenceBatch(pred_score=score, anomaly_map=amap)
```

---

# 6. losses.py (MVTec 파인튜닝용)

MVTec AD는 테스트셋에 anomaly mask가 있으므로, FE-CLIP 파인튜닝을 하려면 보통:

* `L_cls`: 이미지 라벨 BCE
* `L_mask`: focal + dice (세그가 있을 때)

```python
# losses.py
import torch
import torch.nn.functional as F

def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha=0.25, gamma=2.0, eps=1e-6):
    pred = pred.clamp(eps, 1 - eps)
    pt = torch.where(target > 0.5, pred, 1 - pred)
    w = torch.where(target > 0.5, alpha, 1 - alpha)
    return (-w * (1 - pt) ** gamma * torch.log(pt)).mean()

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return (1 - dice).mean()

def bce_loss(pred: torch.Tensor, target: torch.Tensor):
    return F.binary_cross_entropy(pred, target)
```

---

# 7. lightning_model.py (학습 루프 포함)

핵심:

* optimizer는 **requires_grad=True인 파라미터만**

  * adapters(FFE/LFS)
  * fc_patch
* CLIP 본체는 freeze 상태 유지
* training_step에서:

  * pred_score ↔ image label
  * anomaly_map ↔ pixel mask

```python
# lightning_model.py
import torch
from torch import nn
from anomalib.models.components import AnomalibModule
from anomalib.data import Batch

from .torch_model import FEClipModel
from .losses import focal_loss, dice_loss, bce_loss

class FEClip(AnomalibModule):
    def __init__(
        self,
        backbone: str,
        pretrained: str,
        image_size: int = 336,
        n_taps: int = 4,
        lambda_fuse: float = 0.1,
        P: int = 3,
        Q: int = 3,
        lr: float = 5e-4,
        w_cls: float = 1.0,
        w_mask: float = 1.0,
        **kwargs,
    ):
        super().__init__(pre_processor=True, post_processor=True, evaluator=True, visualizer=True)
        self.model = FEClipModel(
            backbone=backbone,
            pretrained=pretrained,
            n_taps=n_taps,
            lambda_fuse=lambda_fuse,
            P=P, Q=Q,
        )
        self.lr = lr
        self.w_cls = w_cls
        self.w_mask = w_mask

    def setup(self, stage: str) -> None:
        del stage
        self.model.setup_text()

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.lr)

    def training_step(self, batch: Batch, *args, **kwargs):
        del args, kwargs

        # batch fields는 datamodule에 따라 다를 수 있음:
        # - batch.image: (B,3,H,W)
        # - batch.gt_label: (B,) 0/1
        # - batch.gt_mask: (B,1,H,W) or (B,H,W)
        out = self.model(batch.image)

        loss = 0.0

        if hasattr(batch, "gt_label") and batch.gt_label is not None:
            y = batch.gt_label.float().view(-1)
            loss_cls = bce_loss(out.pred_score, y)
            loss = loss + self.w_cls * loss_cls
            self.log("train_loss_cls", loss_cls, prog_bar=True)

        if hasattr(batch, "gt_mask") and batch.gt_mask is not None:
            m = batch.gt_mask.float()
            if m.ndim == 4:
                m = m.squeeze(1)
            loss_mask = focal_loss(out.anomaly_map, m) + dice_loss(out.anomaly_map, m)
            loss = loss + self.w_mask * loss_mask
            self.log("train_loss_mask", loss_mask, prog_bar=True)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, *args, **kwargs):
        del args, kwargs
        out = self.model(batch.image)
        return batch.update(pred_score=out.pred_score, anomaly_map=out.anomaly_map)
```

---

# 8. MVTec AD로 “파인튜닝 데이터”를 어떻게 만들까?

여기가 제일 중요해.
MVTec의 공식 split은 **train은 정상만**, test에 정상+이상+mask가 있어.
FE-CLIP 논문 계열은 보통 **라벨/마스크가 있는 쪽으로 adapter를 적응**시키기 때문에, “MVTec 파인튜닝”을 하려면 다음 중 하나를 선택해야 함:

## 옵션 A) Transductive 파인튜닝(빠른 베이스라인)

* **MVTec의 test split을 “adapter 학습용 데이터”로 사용**
* 목적: MVTec으로 adapter가 “결함 주파수”를 익히게 만들기
* 단, 이 모델을 MVTec 자체에 다시 평가하면 누수이므로 **현장 데이터에만 평가**하는 용도로 쓰는 게 깔끔

## 옵션 B) Test split에서 일부만 사용 (더 보수적)

* MVTec test 중 일부(예: defect 이미지의 일부)만 adapter 학습에 사용
* 나머지는 hold-out

---

# 9. MVTec 파인튜닝용 Datamodule/Loader 만들기 (간단 버전)

anomalib의 MVTecAD datamodule은 train이 정상-only라서,
adapter 학습을 위해서는 **test 데이터를 학습용으로 읽는 Dataset**을 하나 만드는 게 편해.

아래는 “최소 구현” 예시(폴더 구조가 표준 MVTec 기준):

```python
# mvtec_finetune_dataset.py (프로젝트 어딘가)
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class MVTecFinetuneDataset(Dataset):
    """
    root/category/test/*/*.png
    root/category/ground_truth/*/*.png
    - good에는 mask 없음 -> mask=0
    - defect에는 mask 존재
    """
    def __init__(self, root: str, category: str, image_size: int):
        self.root = Path(root) / category
        self.image_size = image_size
        self.items = []

        test_dir = self.root / "test"
        gt_dir = self.root / "ground_truth"

        for defect_type in sorted([p.name for p in test_dir.iterdir() if p.is_dir()]):
            for img_path in sorted((test_dir / defect_type).glob("*.png")):
                if defect_type == "good":
                    self.items.append((img_path, None, 0))
                else:
                    # mask name convention: <img>_mask.png in corresponding gt folder
                    mask_dir = gt_dir / defect_type
                    mask_path = mask_dir / (img_path.stem + "_mask.png")
                    self.items.append((img_path, mask_path, 1))

        self.t_img = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])
        self.t_mask = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.items[idx]
        img = self.t_img(Image.open(img_path).convert("RGB"))

        if mask_path is None or not mask_path.exists():
            mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        else:
            mask = self.t_mask(Image.open(mask_path).convert("L"))
            mask = (mask > 0.5).float()

        return {"image": img, "gt_label": torch.tensor(label, dtype=torch.float32), "gt_mask": mask}
```

이 Dataset을 anomalib Batch 형태로 맞추려면 collate를 쓰거나, `Batch` 생성 부분만 맞추면 돼.
(너의 코드베이스에서 `Batch`가 dict를 받아 자동 변환되는지 확인 필요)

---

# 10. “파인튜닝 실행 스크립트” 예시

Lightning/Engine 흐름은 프로젝트에 따라 다르지만, 가장 단순한 PyTorch Lightning 학습 예시는 아래처럼 갑니다.

```python
# train_feclip_mvtec.py (예시)
import torch
from torch.utils.data import DataLoader
import lightning as L

from anomalib.data import Batch
from anomalib.engine import Engine
from anomalib.models.image.feclip.lightning_model import FEClip
from mvtec_finetune_dataset import MVTecFinetuneDataset

def collate_fn(batch_list):
    # dict list -> Batch
    images = torch.stack([b["image"] for b in batch_list])
    labels = torch.stack([b["gt_label"] for b in batch_list]).view(-1)
    masks  = torch.stack([b["gt_mask"] for b in batch_list])  # (B,1,H,W)
    return Batch(image=images, gt_label=labels, gt_mask=masks)

def main():
    root = "./datasets/MVTecAD"
    category = "bottle"
    image_size = 336

    ds = MVTecFinetuneDataset(root, category, image_size=image_size)
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)

    model = FEClip(
        backbone="ViT-L-14-336",
        pretrained="openai",
        image_size=image_size,
        lr=5e-4,
        n_taps=4,
        P=3, Q=3,
        lambda_fuse=0.1,
    )

    # anomalib Engine 사용(프로젝트 버전에 따라 Engine.fit/Engine.train 명칭 다를 수 있음)
    engine = Engine(max_epochs=9, accelerator="gpu", devices=1)
    engine.fit(model=model, train_dataloaders=dl, val_dataloaders=dl)

    # adapter weight 저장 (간단히 전체 state_dict 저장해도 되지만, adapter만 저장하는 게 좋음)
    torch.save(model.state_dict(), "feclip_mvtec_tuned.ckpt")

if __name__ == "__main__":
    main()
```

---

# 11. 현장 데이터 적용(추론)

* 파인튜닝한 checkpoint 로드
* 현장 데이터 datamodule로 `Engine.test()` 또는 `Engine.predict()`

추론 시 특히 중요:

* 너의 현장 데이터는 aspect ratio 유지로 **화이트 padding**이 큼
  → 가능하면 **valid_mask_tokens**를 만들어 LFS에 넣는(위 코드의 `valid_mask_tokens`) 옵션을 켜는 걸 권장
  (이 마스크는 “이미지 유효영역=1, padding=0”을 **토큰 그리드(Ht×Wt)로 다운샘플링**해서 만들면 됨)

---

# 12. 디버깅/검증 체크리스트

1. **open_clip ViT 내부 속성명 확인**

   * `print(model.model.visual)`로 `conv1/class_embedding/positional_embedding/ln_pre/transformer.resblocks/ln_post/proj/grid_size` 존재 확인
2. **FFE P=3에서 Ht,Wt divisible 문제**

   * 위 구현처럼 pad/crop이 되어야 런타임 에러가 안 남
3. 학습 파라미터 확인

   * `sum(p.requires_grad for p in model.parameters())`로 adapter/fc_patch만 학습되는지 체크
4. 출력 shape

   * `pred_score`: (B,)
   * `anomaly_map`: (B,H,W)

---

# 13. 권장 실험 순서(가장 빠른 성공 루트)

1. **추론 먼저**: 파인튜닝 없이 1장 넣어서 forward가 도는지 확인
2. **MVTec로 1~2 epoch만** 돌려서 loss가 내려가는지 확인
3. **9 epoch** 파인튜닝 완료 후 checkpoint 고정
4. 현장 데이터에 적용 + (가능하면) padding mask 버전과 비교

---

## 마지막으로: 너에게 필요한 “결정 2개”

가이드 문서를 네 anomalib 환경에 “딱 맞게” 마무리하려면 아래 2가지만 알려줘.

1. 너가 실제로 쓸 `open_clip` backbone 이름은 뭐로 갈 거야? -> 논문처럼 `ViT-L-14-336`
2. 현장 데이터는 **픽셀 마스크 GT가 없고 이미지 라벨만** 있는 상황이야? -> 맞음.


---

## 1~13에서 수정/보강할 부분

### (수정 1) 4장(Adapters)에서 “FFE의 P=3 divisibility 문제” 톤 다운

* **ViT-L/14@336이면 token grid가 336/14 = 24 → 24×24**라서 **P=3이 딱 나눠떨어짐**.
* 그래서 “14×14에서 P=3 나눗셈 문제”는 너 설정에서는 *원칙적으로* 발생하지 않아.
* 다만 안전하게 유지하려면 지금처럼 pad/crop 코드를 남겨도 되지만, 문서에는:

  * “**ViT-L-14-336은 24×24라 기본 설정(P=3)에서 문제 없음**”
  * “백본 바꾸면 다시 발생 가능하니 pad/crop 유지 권장”
    이렇게 적는 게 더 정확해.

### (수정 2) 7장(lightning_model)에서 “현장 데이터” 전제에 맞게 loss/step 설명 분리

지금 문서는 학습 코드를 “마스크 있을 때 focal+dice 포함”으로 강하게 안내하고 있는데,
너는 **현장 데이터에 mask가 없고**, MVTec에서만 mask를 쓰는 구조잖아.

그래서 문서에서 **학습을 2단계로 명확히 분리**하는 문장이 들어가면 좋아:

* **(A) MVTec 파인튜닝 단계:** `L_cls + L_mask(focal+dice)` 가능 (mask 있음)
* **(B) 현장 적용 단계:** 학습 없음(또는 이미지 라벨만 있으면 `L_cls`만으로 경량 추가 튜닝 가능), 대신 **정상 기반 캘리브레이션/threshold가 핵심**

즉 training_step 예시는 유지하되, “현장에서는 mask loss가 비활성화된다/쓸 수 없다”를 명시.

### (수정 3) 8~9장(MVTec 파인튜닝 데이터)에서 “transductive” 주의 문장 강화

너 목적이 “MVTec으로 튜닝 → 현장 적용”이면 완전 괜찮지만,
문서에 **‘MVTec test를 튜닝에 쓰면 MVTec 성능 보고 시 누수’**를 더 또렷하게 넣는 게 좋아.
(지금도 언급은 있는데, 실험 보고 시 혼란이 많이 생기는 지점이라 강조 추천)

### (수정 4) 11장(현장 적용)에서 “현장에는 mask GT가 없으므로 평가/결정 기준”을 구체화

현장에서는 결국

* **이미지 스코어로 판정**(불량/정상)
* 픽셀 맵은 “설명/가시화” 또는 “후처리 규칙 기반 판정(최대값/상위 p%)”
  이 될 가능성이 높음.

그래서 문서에 “현장 판정 규칙”을 1~2줄이라도 넣어주면 바로 실행 가능해져:

* 예: `pred_score > T_img` 또는 `anomaly_map.max() > T_px` 같은 형태

---

## 13 이후에 “추가하면 좋은 내용” (너 상황에 딱 필요한 것들)

### (추가 A) **정상 기반 캘리브레이션/임계치(Percentile Threshold) 섹션** — 필수

현장에 mask가 없으니, 사실상 이게 구현 완성의 마지막 퍼즐이야. 아래 내용을 문서에 추가 추천:

1. **캘리브레이션 데이터**

* 현장 **정상(train/초기 런칭 기간 수집 정상)** N장을 모음

2. **어떤 스코어를 캘리브레이션할지**

* 이미지: `pred_score` (B,)
* 픽셀: `anomaly_map`에서 하나의 스칼라로 줄여서 사용

  * 추천 1: `amax` (최대값) → 얇은 라인 결함에 민감
  * 추천 2: `topk mean` (상위 k% 평균) → 노이즈에 덜 민감

3. **퍼센타일 선택(p99.5 vs p99.9) 가이드**

* **p99.5**: 민감도↑(검출↑) / 오탐↑
* **p99.9**: 오탐↓ / 미탐↑
* 너는 “얇고 약한 결함”이라서 초반엔 **p99.5로 시작**하고, 운영 오탐이 너무 높으면 p99.7~p99.9로 올리는 흐름이 현실적.

4. **저장해야 할 아티팩트**

* `T_img`, `T_px`(혹은 `T_px_topk`) 값을 **체크포인트와 함께 저장**
* 그래야 현장 적용 재현성이 생김

> 이 섹션이 들어가면 “현장 라벨만” 조건에서 문서가 완결돼.

### (추가 B) **현장용 Score Normalization**(옵션이지만 효과 큼)

퍼센타일 threshold만으로도 되지만, 도메인 시프트가 크면 안정성을 위해 아래를 추가하는 게 좋음:

* 정상 데이터에서 `pred_score`의 평균/표준편차로 z-score 정규화
  또는 min-max/robust scaling(IQR)
* 픽셀 맵도 동일하게 “정상 기준 분포”로 normalize 후 threshold

이건 특히 “라인 결함처럼 약한 신호”에서 threshold가 흔들릴 때 도움이 돼.

### (추가 C) **현장 평가 프로토콜(라벨이 이미지 단위만 있을 때)** — 권장

* AUROC/PR-AUC(이미지 단위)
* 운영 관점이면 “특정 FPR에서의 TPR” 같은 지표도 같이 추천
* 픽셀 맵은 정량 평가가 어려우니: 시각화/inspection 중심으로

### (추가 D) **Letterbox(화이트 패딩) 처리 가이드** — 너 데이터에 중요

이미 앞에서 여러 번 이야기했지만, 문서로 “실행 체크리스트”로 박아두면 좋아:

* 전처리에서 letterbox를 한다면:

  * (1) **유효영역 마스크(valid mask)** 생성
  * (2) 이 마스크를 **token grid(24×24)로 downsample**해서 `valid_mask_tokens`로 LFS에 전달
  * (3) (가능하면) anomaly map 시각화/후처리에서도 padding 영역 무시

---

## 너 설정에서 “한 줄로 문서에 넣으면 좋은 핵심 요약”

* **ViT-L-14-336 → token grid 24×24라 P=3, Q=3 모두 구조적으로 잘 맞음**
* **현장에는 mask GT가 없으므로, 성능 안정화의 핵심은 정상 기반 percentile 캘리브레이션(T_img/T_px)**
* **화이트 패딩이 크면 valid_mask_tokens 기반 LFS masked-mean을 강권**

원하면, 내가 위 “추가 A(캘리브레이션/threshold)” 섹션을 문서 스타일로 그대로 붙여넣을 수 있게 **완성 문단 + 의사코드(몇 줄)** 형태로 만들어줄게.
