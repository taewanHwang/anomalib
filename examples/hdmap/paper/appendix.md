# Appendix

## A. Detailed Model Architecture

This section provides detailed architectural specifications of the models used in our experiments.

### A.1 DRAEM CutPaste (Proposed Method)

Our proposed method combines the DRAEM architecture with CutPaste-based synthetic anomaly generation for unsupervised anomaly detection.

**Architecture Overview:**
- **Reconstructive Subnetwork**: Autoencoder for image reconstruction (1-channel)
- **Discriminative Subnetwork**: U-Net style network for pixel-level anomaly segmentation (2-channel input)
- **Synthetic Generator**: CutPaste-based anomaly synthesis (training only)
- **Anomaly Scoring**: Uses maximum of pixel-wise anomaly map (same as original DRAEM)

**Total Parameters**: 182.48M
- Reconstructive Subnetwork: 69.04M parameters
- Discriminative Subnetwork: 113.44M parameters

#### Reconstructive Subnetwork (69.04M params)

| Layer | Type | Input Channels | Output Channels | Kernel Size | Stride | Output Size |
|-------|------|----------------|-----------------|-------------|--------|-------------|
| **Encoder** |
| Block1 | Conv2d + BN + ReLU | 1 | 128 | 3×3 | 1 | 128×128 |
| Block1 | Conv2d + BN + ReLU | 128 | 128 | 3×3 | 1 | 128×128 |
| MaxPool1 | MaxPool2d | 128 | 128 | 2×2 | 2 | 64×64 |
| Block2 | Conv2d + BN + ReLU | 128 | 256 | 3×3 | 1 | 64×64 |
| Block2 | Conv2d + BN + ReLU | 256 | 256 | 3×3 | 1 | 64×64 |
| MaxPool2 | MaxPool2d | 256 | 256 | 2×2 | 2 | 32×32 |
| Block3 | Conv2d + BN + ReLU | 256 | 512 | 3×3 | 1 | 32×32 |
| Block3 | Conv2d + BN + ReLU | 512 | 512 | 3×3 | 1 | 32×32 |
| MaxPool3 | MaxPool2d | 512 | 512 | 2×2 | 2 | 16×16 |
| Block4 | Conv2d + BN + ReLU | 512 | 1024 | 3×3 | 1 | 16×16 |
| Block4 | Conv2d + BN + ReLU | 1024 | 1024 | 3×3 | 1 | 16×16 |
| MaxPool4 | MaxPool2d | 1024 | 1024 | 2×2 | 2 | 8×8 |
| Block5 | Conv2d + BN + ReLU | 1024 | 1024 | 3×3 | 1 | 8×8 |
| Block5 | Conv2d + BN + ReLU | 1024 | 1024 | 3×3 | 1 | 8×8 |
| **Decoder** |
| Up1 | Upsample + Conv2d + BN + ReLU | 1024 | 1024 | 3×3 | 1 | 16×16 |
| DB1 | Conv2d + BN + ReLU | 1024 | 1024 | 3×3 | 1 | 16×16 |
| DB1 | Conv2d + BN + ReLU | 1024 | 512 | 3×3 | 1 | 16×16 |
| Up2 | Upsample + Conv2d + BN + ReLU | 512 | 512 | 3×3 | 1 | 32×32 |
| DB2 | Conv2d + BN + ReLU | 512 | 512 | 3×3 | 1 | 32×32 |
| DB2 | Conv2d + BN + ReLU | 512 | 256 | 3×3 | 1 | 32×32 |
| Up3 | Upsample + Conv2d + BN + ReLU | 256 | 256 | 3×3 | 1 | 64×64 |
| DB3 | Conv2d + BN + ReLU | 256 | 256 | 3×3 | 1 | 64×64 |
| DB3 | Conv2d + BN + ReLU | 256 | 128 | 3×3 | 1 | 64×64 |
| Up4 | Upsample + Conv2d + BN + ReLU | 128 | 128 | 3×3 | 1 | 128×128 |
| DB4 | Conv2d + BN + ReLU | 128 | 128 | 3×3 | 1 | 128×128 |
| DB4 | Conv2d + BN + ReLU | 128 | 128 | 3×3 | 1 | 128×128 |
| Output | Conv2d | 128 | 1 | 3×3 | 1 | 128×128 |

#### Discriminative Subnetwork (113.44M params)

**Input**: Concatenation of [original image, reconstruction] (2 channels)

| Layer | Type | Input Channels | Output Channels | Kernel Size | Stride | Output Size |
|-------|------|----------------|-----------------|-------------|--------|-------------|
| **Encoder** |
| Block1 | Conv2d + BN + ReLU | 2 | 128 | 3×3 | 1 | 128×128 |
| Block1 | Conv2d + BN + ReLU | 128 | 128 | 3×3 | 1 | 128×128 |
| MaxPool1 | MaxPool2d | 128 | 128 | 2×2 | 2 | 64×64 |
| ... (similar U-Net structure) ... |
| **Decoder with Skip Connections** |
| ... (U-Net decoder) ... |
| Output | Conv2d | 128 | 2 | 3×3 | 1 | 128×128 |

**Output**: Pixel-wise classification logits (2 classes: normal/anomaly)

---

### A.2 CutPaste Clf (Baseline Comparison)

Simple CNN classifier without reconstruction or localization capabilities. This baseline model uses only CutPaste synthetic anomaly generation for training and a shallow CNN for binary classification.

**Total Parameters**: 4.97M

**Architecture Overview:**
- **Feature Extraction**: 2 convolutional layers with max pooling
- **Classification Head**: 3 fully connected layers
- **No Reconstruction**: Unlike DRAEM-based methods, this model does not perform image reconstruction
- **No Localization**: Outputs only image-level classification (no pixel-wise anomaly maps)

#### Detailed Architecture

| Layer | Type | Input Channels | Output Channels | Kernel Size | Other Params | Output Size | Params |
|-------|------|----------------|-----------------|-------------|--------------|-------------|---------|
| **Feature Extraction** |
| Conv1 | Conv2d | 3 | 32 | 5×5 | padding=2 | 128×128 | 2,432 |
| ReLU1 | ReLU | - | - | - | - | 128×128 | 0 |
| MaxPool1 | MaxPool2d | 32 | 32 | 3×3 | stride=2, pad=1 | 64×64 | 0 |
| Conv2 | Conv2d | 32 | 48 | 5×5 | padding=2 | 64×64 | 38,448 |
| ReLU2 | ReLU | - | - | - | - | 64×64 | 0 |
| MaxPool2 | MaxPool2d | 48 | 48 | 3×3 | stride=2, pad=1 | 32×32 | 0 |
| **Classification Head** |
| Flatten | Flatten | - | - | - | - | 49,152 | 0 |
| FC1 | Linear | 49,152 | 100 | - | - | 100 | 4,915,300 |
| ReLU3 | ReLU | - | - | - | - | 100 | 0 |
| FC2 | Linear | 100 | 100 | - | - | 100 | 10,100 |
| ReLU4 | ReLU | - | - | - | - | 100 | 0 |
| FC3 | Linear | 100 | 2 | - | - | 2 | 202 |

**Parameter Breakdown:**
- Convolutional Layers: 40,880 (0.04M) - 0.8% of total
- Fully Connected Layers: 4,925,602 (4.93M) - 99.2% of total
- Total: 4,966,482 (4.97M)

**Key Characteristics:**
- **Lightweight**: ~97% fewer parameters than DRAEM CutPaste
- **Fast**: Shorter training and inference time
- **Image-Level Only**: No pixel-wise anomaly localization
- **3-Channel Input**: Grayscale images are replicated across RGB channels

**Input**: 3-channel RGB image (128×128) - grayscale replicated to 3 channels
**Output**: Binary classification logits (2 classes: normal/anomaly)

---

### A.3 DRAEM (Original)

Original DRAEM architecture with Perlin noise-based anomaly synthesis.

**Total Parameters**: 97.42M
- Reconstructive Subnetwork: 69.05M parameters (3-channel version)
- Discriminative Subnetwork: 28.37M parameters (6-channel input)

**Differences from our method:**
- Uses 3-channel RGB input
- Perlin noise-based synthetic anomaly generation
- No classification head (only pixel-level segmentation)
- Discriminative network input: 6 channels (3 original + 3 reconstruction)

---

## B. Training Configuration

This section details the training hyperparameters and configurations used across all experiments.

### B.1 Common Training Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 128 × 128 | Input image resolution |
| Optimizer | AdamW | Adaptive learning rate optimizer |
| Weight Decay | 0.01 | L2 regularization strength |
| Validation Split | 0.1 | 10% of training data for validation |
| Seed | 52 | Random seed for reproducibility |
| Device | NVIDIA GPU | Training hardware |
| Precision | FP32 | Floating point precision |

---

### B.2 DRAEM CutPaste (Proposed) - Training Configuration

**Model-Specific Settings:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max Epochs | 3 | Maximum training epochs |
| Batch Size | 64 | Training batch size |
| Learning Rate | 5×10⁻⁴ | Initial learning rate |
| Early Stopping | 20 epochs | Patience for early stopping |

**Learning Rate Scheduler:**
- Type: StepLR
- Step Size: 1 epoch
- Gamma: 0.1 (LR decay factor)

**CutPaste Augmentation Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| cut_w_range | [2, 127] | Patch width range (pixels) |
| cut_h_range | [4, 8] | Patch height range (pixels) |
| a_fault_start | 0.0 | Minimum fault amplitude |
| a_fault_range_end | 0.1/0.2/0.3* | Maximum fault amplitude (domain-dependent) |
| augment_probability | 0.5 | Probability of applying augmentation |
| norm | True | Apply normalization to synthetic faults |

*Fault severity varies by experiment: 0.1 (10%), 0.2 (20%), 0.3 (30%)

**Loss Functions:**

| Loss Component | Weight | Description |
|----------------|--------|-------------|
| L2 Loss (Reconstruction) | 1.0 | MSE between original and reconstructed images |
| SSIM Loss (Reconstruction) | 1.0 | Structural similarity loss |
| Focal Loss (Segmentation) | α=0.1 | Pixel-level anomaly segmentation |

**Note**: Unlike DRAEM CutPaste Clf variant, this model does not use image-level classification loss. Anomaly detection is performed using the maximum value of the pixel-wise anomaly map (same as original DRAEM).

---

### B.3 CutPaste Clf (Baseline) - Training Configuration

**Model-Specific Settings:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max Epochs | 3 | Maximum training epochs |
| Batch Size | 128 | Training batch size (larger than DRAEM) |
| Learning Rate | 5×10⁻⁴ | Initial learning rate |
| Early Stopping | 20 epochs | Patience for early stopping |

**Learning Rate Scheduler:**
- Type: StepLR
- Step Size: 10 epochs
- Gamma: 0.8

**CutPaste Augmentation:** Same as DRAEM CutPaste (Proposed)

**Loss Function:**
- Cross-Entropy Loss: Image-level binary classification only

---

### B.4 DRAEM (Original) - Training Configuration

**Model-Specific Settings:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max Epochs | 3 | Maximum training epochs |
| Batch Size | 32 | Training batch size (smaller due to 3-channel) |
| Learning Rate | 1×10⁻⁴ | Initial learning rate (lower) |
| Early Stopping | 20 epochs | Patience for early stopping |

**Learning Rate Scheduler:**
- Type: StepLR
- Step Size: 10 epochs
- Gamma: 0.8

**Synthetic Anomaly Generation:**
- Method: Perlin Noise
- No explicit parameters (probabilistic generation)

**Loss Functions:**

| Loss Component | Weight | Description |
|----------------|--------|-------------|
| L2 Loss | 1.0 | Reconstruction loss |
| SSIM Loss | 1.0 | Structural similarity loss |
| Focal Loss | default | Segmentation loss |

---

### B.5 Other Baseline Methods

#### PaDiM
- Embedding Network: ResNet18 (pretrained on ImageNet)
- Feature Extraction Layers: Layer 1, 2, 3
- Dimensionality Reduction: Random projection (100 dimensions)
- Anomaly Detection: Mahalanobis distance

#### FastFlow
- Backbone: ResNet18 (pretrained on ImageNet)
- Flow Model: 2D normalizing flow
- Training Epochs: 500
- Batch Size: 32
- Learning Rate: 1×10⁻³

#### Reverse Distillation
- Teacher: Wide ResNet-50-2 (pretrained)
- Student: ResNet-18
- Distillation Loss: MSE on multi-scale features
- Training Epochs: 200
- Batch Size: 32

---

## C. Dataset Configuration

### C.1 Health Data Map (HDMAP) Dataset

**Dataset Statistics:**

| Domain | Sensor Type | Data Type | Training Samples | Test Normal | Test Fault | Total |
|--------|-------------|-----------|------------------|-------------|------------|-------|
| A | Class1/1 | 3_TSA_DIF | 100,000 | 2,000 | 2,000 | 104,000 |
| B | Class1/1 | 1_TSA_DIF | 100,000 | 2,000 | 2,000 | 104,000 |
| C | Class3/1 | 3_TSA_DIF | 100,000 | 2,000 | 2,000 | 104,000 |
| D | Class3/1 | 1_TSA_DIF | 100,000 | 2,000 | 2,000 | 104,000 |

**Preprocessing:**

1. **Original Data Format**: MATLAB .mat files (grayscale health data maps)
2. **Conversion**: Converted to 32-bit float TIFF format
3. **Normalization**: Min-max normalization using domain-specific ranges

| Domain | user_min | user_max | Description |
|--------|----------|----------|-------------|
| A | 0.0 | 0.324670 | Normalized to [0, 1] range |
| B | 0.0 | 1.324418 | Normalized to [0, 1] range |
| C | 0.0 | 0.087341 | Normalized to [0, 1] range |
| D | 0.0 | 0.418999 | Normalized to [0, 1] range |

4. **Resizing**: Bilinear interpolation to 128×128 pixels
5. **Channel Configuration**:
   - DRAEM CutPaste (Proposed): Uses 1-channel (grayscale)
   - CutPaste Clf (Baseline): Replicates to 3-channel
   - DRAEM (Original): Replicates to 3-channel

**Data Split:**
- Training: 90% of normal samples (90,000 images)
- Validation: 10% of normal samples (10,000 images)
- Test: All test samples (2,000 normal + 2,000 fault = 4,000 images)

---

## D. Evaluation Metrics

All models are evaluated using the following metrics:

**Image-Level Metrics:**
- **AUROC**: Area Under the Receiver Operating Characteristic curve
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

**Pixel-Level Metrics** (for models with localization):
- **Pixel AUROC**: AUROC computed at pixel level
- **PRO**: Per-Region Overlap score

**Threshold Selection:**
- Optimal threshold determined on validation set using F1-score maximization
- Same threshold applied to all test domains

---

## E. Computational Requirements

**Training Time** (approximate, on NVIDIA V100 32GB):

| Model | Training Time per Epoch | Total Training Time (3 epochs) |
|-------|-------------------------|--------------------------------|
| DRAEM CutPaste (Proposed) | ~40 min | ~2.0 hours |
| CutPaste Clf (Baseline) | ~15 min | ~0.75 hours |
| DRAEM (Original) | ~60 min | ~3.0 hours |
| PaDiM | N/A (no training) | ~5 min (feature extraction) |
| FastFlow | ~10 min | ~1.5 hours (500 epochs) |

**GPU Memory Usage** (batch size as configured):

| Model | Batch Size | Peak GPU Memory |
|-------|------------|-----------------|
| DRAEM CutPaste (Proposed) | 64 | ~24 GB |
| CutPaste Clf (Baseline) | 128 | ~8 GB |
| DRAEM (Original) | 32 | ~24 GB |

---

## F. Implementation Details

**Framework:**
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- Anomalib (custom modifications)

**Key Modifications:**
1. Added 1-channel optimization for DRAEM
2. Implemented CutPaste-based synthetic anomaly generator
3. Modified discriminative network base width to 128 (for 1-channel optimization)
4. Integrated CutPaste augmentation with DRAEM's reconstruction-based approach

**Code Availability:**
- Source code: `src/anomalib/models/image/draem_cutpaste/`
- Baseline code: `src/anomalib/models/image/cutpaste_clf/`
- Experiment configurations: `examples/hdmap/single_domain/`
- Visualization tools: `examples/hdmap/paper/`

---

## G. Hyperparameter Sensitivity Analysis

**CutPaste Severity (a_fault_range_end):**

We evaluated three severity levels: 0.1 (10%), 0.2 (20%), and 0.3 (30%)

- **Lower severity (0.1)**: Better for subtle anomalies but may miss severe faults
- **Higher severity (0.3)**: Better for severe anomalies but may overfit to synthetic patterns
- **Optimal**: Domain-dependent, typically 0.2-0.3 for health data maps

**Focal Loss Alpha:**

Tested values: [0.05, 0.1, 0.25, 0.5]

- **Lower alpha (0.05)**: More balanced between normal and anomaly classes
- **Higher alpha (0.5)**: Focuses more on anomaly class
- **Optimal**: 0.1 provides best balance for our dataset

**Learning Rate:**

Tested values: [1e-4, 5e-4, 1e-3]

- **Lower LR (1e-4)**: More stable but slower convergence
- **Higher LR (1e-3)**: Faster convergence but may overshoot
- **Optimal**: 5e-4 balances speed and stability

---

## References

1. DRAEM: Zavrtanik et al., "DRAEM - A discriminatively trained reconstruction embedding for surface anomaly detection," ICCV 2021
2. CutPaste: Li et al., "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization," CVPR 2021
3. PaDiM: Defard et al., "PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization," ICPR 2021
4. FastFlow: Yu et al., "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows," arXiv 2021
5. Reverse Distillation: Deng & Li, "Anomaly Detection via Reverse Distillation from One-Class Embedding," CVPR 2022
