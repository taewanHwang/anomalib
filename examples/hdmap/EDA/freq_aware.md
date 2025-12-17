
# FAIR: Frequency-aware Image Restoration

### (핵심 맥락 요약 – 2D FFT 기반 차별점 중심)

## 문제의식 (왜 Frequency인가?)

기존 **재구성 기반 anomaly detection**은 다음의 구조적 trade-off를 가짐:

* **정상 재구성 품질 ↑**
  → skip connection, over-parameterization
  → 이상 영역까지 identity mapping → anomaly 구분력 ↓
* **이상 구분력 ↑**
  → latent 압축
  → 정상 디테일 손실 → false positive ↑

이 논문은 이 trade-off가 **공간 도메인 문제가 아니라, 주파수 도메인 문제**라고 재정의함.

---

## 핵심 관찰 (Frequency Bias)

논문의 핵심 인사이트는 다음과 같음 :

* **정상 재구성 오류**

  * 다운샘플링 + MSE loss 영향
  * **고주파(high-frequency)에 편향**
* **이상 재구성 오류**

  * anomaly의 불확실성으로 인해
  * **전 주파수 대역(low~high)에 분포**

즉,

> 정상/이상 간 차이는 “얼마나 틀렸는가”가 아니라
> **“어느 주파수에서 틀렸는가”**에 있음

---

## 핵심 아이디어: 2D FFT 기반 Restoration Task

### 1. 입력을 “고주파만 남긴 이미지”로 제한

* 입력 이미지에 **2D Discrete Fourier Transform (DFT)** 적용
* 저주파 성분 제거 → 고주파 성분만 유지

수식적으로:

```math
F(u,v) = \mathcal{F}\{f(x,y)\}
```

```math
F_h(u,v) = F(u,v) \cdot H_{HPF}(u,v)
```

```math
f_h(x,y) = \mathcal{F}^{-1}\{F_h(u,v)\}
```

여기서 ( H_{HPF} ) 는 **High-Pass Filter**

---

### 2. 왜 High-Pass인가?

* 저주파:

  * 이미지 에너지 대부분 차지
  * anomaly identity mapping의 주범
* 고주파:

  * 정상 재구성 오류가 집중되는 영역
  * anomaly와 정상 간 **frequency bias가 가장 크게 드러남**

👉 **저주파 제거 = anomaly 정보 전달 경로 차단**

---

## 필터 설계의 핵심 (단순 FFT 아님)

논문은 **“어떤 High-pass filter를 쓰느냐”가 성능을 좌우**한다고 분석함.

### 비교된 필터들

* **IHPF (Ideal HPF)**

  * 저주파 완전 제거
  * ❌ ringing artifact 심함 → 정상 재구성 악화
* **GHPF (Gaussian HPF)**

  * 부드러운 감쇠
  * ❌ 저주파 정보 과다 보존 → anomaly identity mapping
* **BHPF (Butterworth HPF)**

  * **타협점**
  * ringing 완화 + 저주파 억제

👉 **2nd-order Butterworth HPF (2-BHPF)**가 최적 

---

## 중요한 결과: Trade-off가 구조적으로 해소됨

이 접근의 핵심 효과:

| 항목     | 기존 Reconstruction | FAIR          |
| ------ | ----------------- | ------------- |
| 정상 재구성 | 디테일 손실 or overfit | 고주파 기반 정밀 재구성 |
| 이상 재구성 | identity mapping  | 저주파 차단으로 억제   |
| 구분 기준  | 공간 오차 크기          | **주파수 분포 차이** |

즉,

> **“얼마나 다르게 재구성되는가” → “어떤 주파수에서 다르게 재구성되는가”**

로 문제 정의 자체가 바뀜.

---

## 왜 단순 UNet으로도 SOTA가 나오는가?

* 복잡한 메모리 뱅크, teacher-student 구조 불필요
* discriminative power가 **아키텍처가 아니라 입력 정보 설계에서 발생**
* FFT 기반 입력 제한이 **암묵적 regularizer 역할**

👉 모델이 “이상까지 잘 재구성할 기회”를 원천적으로 박탈

---

## 추가적으로 중요한 확장 포인트

* FFT 대신 **Sobel gradient (공간 미분)**도 가능

  * gradient = 고주파 성분
  * 방향성 있는 주파수 제거 가능
* 이는 **frequency-domain 설계의 일반화 가능성**을 보여줌

---

## 한 줄 요약 (Agent 전달용)

> FAIR는 anomaly detection에서의 재구성 trade-off를
> **2D FFT 기반 고주파 복원 문제로 재정의**하고,
> 정상/이상 간 **frequency bias 차이**를 활용해
> **저주파를 제거함으로써 anomaly identity mapping을 구조적으로 차단**한 방법이다.

---

원하시면 다음도 도와줄 수 있어요:

* 🔹 이 논문을 **DRAEM / DSR / PatchCore 관점에서 비교 요약**
* 🔹 “왜 low-frequency가 anomaly identity mapping에 치명적인지” 이론적으로 정리
* 🔹 이 아이디어를 **Video / 3D / Multimodal AD**로 확장하는 관점

다음으로 어떤 관점이 필요하신가요?
