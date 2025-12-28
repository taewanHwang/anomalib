# 0) 너의 제안을 한 문장으로 정리

테스트 이미지 (x)에 대해:

1. **Condition gating(이미지 레벨)**:
   cold bank 대표 reference들과의 “전역 유사도” vs warm bank 대표 reference들과의 “전역 유사도”를 비교해서
   (\hat{c}(x)\in{\text{cold},\text{warm}})를 선택

2. **Anomaly scoring(패치 레벨)**:
   선택된 bank (R_{\hat{c}(x)}) 안에서만 WinCLIP reference association으로 anomaly map/score 계산

네가 원한 “cold는 cold bank로 강제”가 (2)에서 성립하고,
(1)에서 bank 선택이 **reference 유사도**로 결정되니 “intensity threshold” 같은 외부 룰보다 더 정합적인 느낌이 나.

---

# 1) Condition-aware 방법 설계: 가장 간단한 binary gating부터

## 방법 CA-1: **Prototype similarity gating (가장 간단, 강추)**

각 condition bank에서 “대표(reference) feature”를 하나의 프로토타입으로 만든 뒤, 테스트와 비교해 bank를 고르는 방식.

### 단계

* cold bank의 정상 reference 이미지들에서 **global feature**를 뽑아 평균:
  [
  p_{cold}=\frac{1}{|B_{cold}|}\sum_{r\in B_{cold}} g(r)
  ]
* warm도:
  [
  p_{warm}=\frac{1}{|B_{warm}|}\sum_{r\in B_{warm}} g(r)
  ]
* 테스트:
  [
  c(x)=\arg\max_{c\in{cold,warm}} \cos(g(x), p_{c})
  ]
  여기서 (g(\cdot))는 “이미지 한 장을 대표하는 벡터(global embedding)”.

### 네가 걱정한 “feature gating”과 뭐가 다르냐?

너는 “학습된 다른 feature로 gating”을 싫어하는 거고, 지금 제안은

* **WinCLIP이 이미 쓰는 같은 임베딩 계열(=CLIP 이미지 인코더)**에서 뽑은 global feature로,
* **reference와의 유사도**만 보는 거야.

즉 “별도 feature”가 아니라, **reference association 철학의 연장선**이야.

> 그리고 중요한 점: 이 방식은 patch-level NN을 돌리기 전에
> global 1번 비교만 하면 돼서 계산도 싸.

---

## 방법 CA-2: **Top-K reference similarity gating (프로토타입보다 더 견고)**

프로토타입 평균 대신, bank 내 여러 reference와의 유사도를 보고 결정:

* cold score:
  [
  S_{cold}(x)=\text{mean of top-K}{\cos(g(x), g(r))\mid r\in B_{cold}}
  ]
* warm도 동일
* (\hat{c}(x)=\arg\max(S_{cold}, S_{warm}))

**장점**

* bank 내부 다양성이 크거나(도메인 A~D가 섞임), 프로토타입 평균이 흐려지는 문제에 더 강함
* K=5 같은 소수로도 충분히 견고해질 때가 많음

---

## 방법 CA-3: **Patch-vote gating (네 아이디어를 가장 ‘정직하게’ 구현)**

네가 말한 것처럼 “이미지의 패치들이 같은 condition bank 패치들과 전반적으로 더 잘 맞을 것”을 직접 쓰는 방법.

* 테스트 패치 (F_x={f_i})
* cold bank 패치 집합 (R_{cold}), warm (R_{warm})

패치마다 “어느 bank가 더 가까운지” 투표해서 결정:
[
vote_i = \mathbb{1}\left[\max_{r\in R_{cold}}\cos(f_i,r);>;\max_{r\in R_{warm}}\cos(f_i,r)\right]
]
[
\hat{c}(x)=\text{majority vote}(vote_i)
]

**장점**

* 너의 직관을 가장 직접적으로 반영
* intensity 같은 단순 통계가 아니라 “패턴/텍스처 유사도”로 condition을 결정

**단점**

* 계산량이 늘 수 있음(하지만 bank를 작게 시작하면 가능)
* anomaly 패치가 많으면 gating이 흔들릴 수 있음(그래서 아래 보완이 필요)

### 보완(중요): gating에는 “상위 정상스러워 보이는 패치만” 쓰기

anomaly가 섞이면 gating이 헷갈릴 수 있어.
그래서 gating에서는

* 패치의 에너지/신뢰도 기준으로 상위 Q%만 쓰거나,
* 또는 “두 bank 모두에서 유사도가 큰 패치만” 투표에 포함

이건 논문화 포인트가 되기도 해: **“gating을 anomaly에 robust하게 만드는 selection”**.

---

# 2) 논문으로 이어질 만한 “Condition-aware WinCLIP” 틀

네가 원하는 방향을 논문처럼 쓰면 아래 한 문장으로 정리돼:

> **Condition-aware reference association**:
> We decouple *condition gating* (image-level bank selection) from *anomaly scoring* (patch-level reference association) and restrict patch matching within the predicted operating condition to mitigate amplitude/scale confounds.

그리고 ablation을 이렇게 잡으면 깔끔해:

* Baseline: WinCLIP (mixed reference / or no reference)
* Oracle: condition 라벨로 bank 고정 선택
* Proposed: gating으로 bank 선택 + 해당 bank에서만 scoring
* Proposed+robust gating: patch-vote + top-K + confidence filtering

---

# 3) “일단 간단하게 binary”로 가자 — 내가 추천하는 최소 구현 버전

너가 말한 것처럼 처음엔 복잡하게 가지 말고, **CA-1 또는 CA-2**부터가 좋아.

### 내가 추천하는 1순위 조합

* **gating: CA-2 (top-K global similarity)**
* **scoring: WinCLIP 그대로(선택된 bank만)**

이유:

* CA-2는 구현이 매우 쉽고(프로토타입 평균보다 안전)
* patch-vote(CA-3)는 계산량/취급 이슈가 있어 2단계로 미루는 게 좋아

---

# 4) 실험 설계(베이스라인을 어떻게 깔끔하게 잡나)

너가 고민하던 “no ref / 1 ref / 2 ref”를 condition-aware에 맞게 재정의해보면:

## Reference budget을 “공정하게” 맞추는 게 핵심

예: 총 reference 수를 N으로 고정하고 비교.

### Baselines

1. **No reference (k=0)**: 지금의 zero-shot
2. **Mixed reference (총 N)**: cold+warm 섞어서 N장
3. **Min-score (총 N + 총 N)**: 두 bank 모두 써서 최소 선택(비추천이지만 비교용)

### Upper bound

4. **Oracle condition-aware (cold N/2, warm N/2)**

   * 테스트 condition을 알고 해당 bank만 씀
   * bank별 reference 수를 N/2로 맞춰 “총 예산” 공정성 유지

### Proposed

5. **Gated condition-aware (cold N/2, warm N/2)**

   * gating으로 bank 선택 후 해당 bank만 scoring

이 설계가 좋은 이유:

* “reference 수가 많아져서 좋아진 건지” vs “condition-aware라서 좋아진 건지”를 분리할 수 있음.

---

# 5) 너의 질문 2번에 대한 답: “binary / n개 선택 / weight” 확장

너가 말한 확장도 전부 자연스럽게 이어져:

* **binary 선택**: bank 하나만 선택 (첫 실험 추천)
* **n개 선택**: 예를 들어 도메인 A/B/C/D까지 bank를 늘렸을 때 top-2 bank만 사용
* **soft weight**: (score = w \cdot Score_{cold} + (1-w)\cdot Score_{warm})

하지만 지금 단계에선 네 말대로 **binary**가 최고야.
왜냐면 soft weight는 다시 “잘못된 bank가 섞여 들어오는” 문제를 부활시킬 수 있어서, 먼저 binary로 “상태를 강제”하는 효과를 확인해야 해.

---

# 6) 다음에 고민할 “실패 모드”까지 미리 짚어보기

Condition-aware가 실패할 수 있는 케이스는 딱 2가지야:

1. **gating 자체가 틀리는 경우**
2. condition은 맞췄는데 **bank 내부 다양성이 부족**해서 nearest가 안 잡히는 경우
   → (해결) bank를 늘리거나, bank를 도메인별로 더 쪼개거나, top-K reference 정책

따라서 실험에서 꼭 같이 기록해야 할 건:

* gating accuracy(oracle 대비 얼마나 틀리는지)
* gating margin(두 bank 유사도 차이)
* bank 크기 vs 성능 곡선

이게 그대로 논문 그림이 돼.

---

## 제안: 다음 실험을 “딱 3개”만 먼저 하자

1. **Oracle condition-aware (N/2+N/2)**
2. **Gated condition-aware (CA-2) (N/2+N/2)**
3. **Mixed reference (N)**

여기서 2)가 1)에 근접하면 성공이고,
3)보다 유의미하게 좋으면 “condition-aware의 가치”가 입증돼.

---

원하면 내가 바로 더 구체화해줄게.
지금 너의 WinCLIP 구현에서 “reference bank”가 실제로는 **이미지 단위로 주어지나**, 아니면 **패치 feature를 미리 캐싱해두나**에 따라 구현 위치가 달라져.

* reference가 이미지 목록이라면: gating은 이미지-이미지 유사도만 추가하면 되고,
* feature 캐시라면: gating도 feature 캐시 기반으로 깔끔하게 가능.

너 지금 anomalib 쪽에서 reference를 어떤 형태로 관리하고 있어? (이미지 파일 목록 vs 캐시된 feature bank)
