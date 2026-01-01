지금까지 분석한 내용을 정리해보면 다음과 같음. 

* **CLIP/WinCLIP도, DINOv2+PatchCore도 “미세한 가로 결함(얇은 band)”은 patch 단위로는 정상과 거의 구분 못 함**
* 게다가 CA-memory bank는 “조건 분리”가 도움이 되기보단 **정상 다양성(coverage)**를 깎아서 오히려 손해가 났고,
* 특히 중요한 관찰: **Cold fault의 ‘의심 패치’가 cold normal보다 warm normal에 더 가까운 경우가 많다** → “조건”으로 쪼개는 방식이 근본 해결이 아님

그래서 네가 제안한 방향(“patch score 계산 단계에서 주변(특히 horizontal) 정보를 반영”)은 **맞는 다음 스텝**이고, 단순 pooling/후처리 smoothing보다 더 근본적인 해결이 가능해.

아래는 **PatchCore + DINOv2를 유지하면서**, “점수 계산 자체”를 공간적으로 바꾸는 아이디어들을 난이도/효과/논문화 포인트까지 포함해서 정리해볼게.

---

## 0) 왜 “pooling만”으로는 안 잡히는가 

지금 상황은 대체로 이런 구조야:

* 각 패치의 원시 거리 (d_{i,j}) 자체가 **정상과 거의 같음** (SNR이 너무 낮음)
* 그런데 사람은 “가로로 길게 이어지는 약한 신호”를 **공간적으로 적분**해서 결함으로 봄
* 따라서 **(사후) d-map을 평균내는 수준**은 신호가 너무 약해 효과가 제한적이고,
* “애초에 측정치 (d_{i,j})를 만들 때” 주변을 같이 보게 만들면, 그때부터 SNR이 올라갈 수 있어.

즉 네 말대로 **“계산 과정에서의 공간 prior”**가 필요해.

---

## 1) 1순위 추천: Contextual Patch Descriptor (CPD-PatchCore)

### 핵심 아이디어

PatchCore의 kNN 입력을 “패치 하나의 feature”가 아니라
**‘중심 패치 + 가로 이웃 패치들’의 컨텍스트 특징**으로 바꿔서, kNN 거리 자체가 “라인 패턴”에 민감해지게 만들자.

#### 기본형 (가장 단순/효과 좋음)

DINO patch feature가 (f_{i,j})일 때:

* 가로 컨텍스트 평균:
  [
  c_{i,j} = \text{mean}{ f_{i,j-k},\dots,f_{i,j+k} }
  ]
* 최종 descriptor:
  [
  g_{i,j} = [f_{i,j} \ | \ c_{i,j}]
  ]
  (그냥 concat)

이러면 “얇은 가로 결함”이 존재할 때:

* 개별 (f_{i,j})는 정상과 비슷해도,
* 같은 y에서 가로로 연속된 패치들이 **같은 방향으로 아주 미세하게 흔들리는 패턴**이 생기면,
* (c_{i,j})가 그 흔들림을 **적분**해서 kNN 거리에서 더 잘 드러날 가능성이 커짐.

#### 왜 이게 “pooling 후처리”보다 낫나?

* pooling 후처리는 “이미 계산된 거리 (d)”를 평균냄
* CPD는 “거리 계산에 들어가는 feature”가 바뀌니까,
  **정상과의 최근접 이웃 자체가 달라지고** 분리도가 커질 수 있어.

#### 구현 난이도 (anomalib 기준)

중간. 하지만 깔끔해:

* `extract_features()`에서 patch tokens를 얻은 뒤,
* 메모리뱅크 저장 전에 한 번 **horizontal conv(평균 필터)** 돌리고 concat만 하면 됨.
* kNN/coreset 로직은 그대로 재사용 가능.

#### 팁(너 데이터 특화)

* 원본이 31×95라서 결함은 “행(row) 단위” 특성이 강함
  → k는 작은 값부터(예: 1,2,3) 시작하고, **(1×k) 비등방 컨텍스트**만 쓰는 게 맞음.

---

## 2) 2순위 추천: “Residual Context” (주변 대비 잔차) 기반 PatchCore

CPD의 변형인데, 미세 결함에서 특히 강할 수 있어.

### 핵심 아이디어

결함이 “가로로 얇은 band”면, 그 행에서 패치들이 전체적으로 비슷하게 움직일 수 있어.
그럴 때는 “절대 feature”보다 **주변 대비 잔차(residual)**가 더 민감할 때가 많아.

[
r_{i,j} = f_{i,j} - \text{mean}{f_{i,j-k},\dots,f_{i,j+k}}
]
그리고 kNN은 (r_{i,j})로 수행하거나, (g=[f|r])로 수행.

이 방식의 장점:

* “전역/상태 변화(예: warm-up 구조 변화)” 성분은 이웃 평균에서 상당 부분 제거되고,
* “좁은 band로 인해 생기는 국소적인 위상/텍스처 변화”가 강조될 수 있음

주의:

* 너무 약한 결함 + 노이즈가 큰 경우엔 residual이 노이즈를 키울 수 있어
  → k를 적절히 두고, 또는 **robust mean(중앙값, trimmed mean)** 같은 걸 쓰면 안정화됨.

---

## 3) 3순위: kNN 자체를 “Row-structured matching”으로 바꾸기 (논문화 포인트 강함)

지금 PatchCore는 각 query patch가 memory bank의 “어느 patch든” 가까운 걸 찾지?
그런데 너 결함은 “특정 y 위치의 얇은 라인”이라서 **row 구조**가 중요해.

### 아이디어 A: Row prototype bank

정상 이미지들에서 각 row의 평균 patch feature를 만들어서,

* 정상 row prototype:
  [
  p_i = \text{mean}*j f*{i,j}
  ]
  테스트에서:
* 각 row의 prototype과 distance를 계산하고
* “row가 이상한지”를 먼저 판단(약한 라인은 row-level에서 더 잘 뜨는 경우가 있음)
* 그 다음 row 내 patch score를 재가중치

이건 완전한 “후처리 pooling”이 아니라,

* **점수의 구조(패치→행→이미지)를 바꿔서**, 약한 band를 “row anomaly”로 증폭시키는 방식이라 꽤 강함.

### 아이디어 B: kNN 후보를 “row-wise 제한/가중”하기

즉, query patch가 ((i,j))면, memory bank에서도 “비슷한 row 성격을 가진 것”을 더 선호하게.

예:

* row-embedding (p_i)가 비슷한 normal patches에 높은 prior weight
* 최종 거리를
  [
  d(f_{i,j}, r) + \lambda \cdot d(p_i, p(r))
  ]
  처럼 만들면 “row context가 같은 정상”을 더 찾게 돼서 미세한 row anomaly를 잘 드러낼 수 있음.

이건 논문화 시 “**structured nearest neighbor with row prior**”로 포장 가능.

---

## 4) 4순위(하지만 강력): 거리맵을 “MRF/CRF 추론”으로 만들기 (단순 smoothing 아님)

너가 “단순 pooling으로는 안 될 듯”이라고 한 건 맞는데,
CRF/MRF는 단순 smoothing이 아니라 **‘약한 관측 + 강한 공간 prior’로 latent anomaly를 추론**하는 거라 케이스에 따라 게임체인저가 될 수 있어.

### 형태

* 관측치: PatchCore distance (d_{i,j}) (노이즈 큼)
* 잠재변수: (z_{i,j}\in{0,1}) (이상 여부)
* prior: 같은 row에서 인접한 패치들은 같은 label일 가능성이 큼(특히 horizontal)

즉, “가로 방향으로 연속된 이상”을 선호하는 에너지 함수로 MAP 추론하면

* 약한 결함이 “연속성” 덕분에 살아날 수 있어.

다만 단점:

* 구현/튜닝이 상대적으로 번거롭고,
* “진짜 모델 개선”이라기보단 “추론/후처리 개선”으로 보일 수 있어
  → 논문화할 땐 “line-structured prior inference”로 잘 포장해야 함.

---

## 5) 너의 실험 결과를 반영한 중요한 결론: “Hard condition gating은 버리고, Soft diversity는 살려라”

CA-PatchCore에서 oracle조차 떨어진 이유가 정말 핵심이야:

* **결함 패치의 최근접 정상 이웃이 ‘다른 condition’에 있을 수 있음**
* 그러면 “cold-only bank”는 오히려 결함을 정상처럼 만들 수 있음

따라서 앞으로의 방향은:

* condition을 분리(하드)하기보단,
* **다양한 정상 manifold를 함께 갖고 있으면서도**,
  “매칭/스코어링”에서 **공간 구조(가로 연속성)**를 반영하는 쪽이 맞아.

즉, 지금 네가 제안한 “spatially-aware scoring”이 CA보다 우선순위가 높아.