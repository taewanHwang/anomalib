# Custom DRAEM for HDMAP Datasets

Custom DRAEM is an extended version of the original DRAEM model with **DRAEM backbone integration** and **Fault Severity Prediction Sub-Network**, specifically designed for HDMAP (Health Data Map) datasets. 

## 🚀 **Latest Update: Direct Comparison Architecture**

### ✨ **Major Breakthrough: Fair Comparison Achievement**
- **DRAEM Backbone Integration**: 97.4M parameters (identical to original DRAEM)
- **Wide ResNet Encoder**: ImageNet pretrained encoder (same as original DRAEM)
- **Fair Comparison**: Pure custom feature effects measured with identical baseline

## 🎯 Key Features

### 🏗️ Architecture (Updated)
- **DRAEM Backbone (97.4M params)**: Original DRAEM's reconstructive + discriminative subnetworks
- **Fault Severity Sub-Network (+118K params)**: Custom continuous severity prediction  
- **3-Channel RGB Support**: Direct processing of NxN RGB images
- **SSPCAB Option**: Optional Self-Supervised Perceptual Consistency Attention Block

### 🔧 Synthetic Fault Generation (Enhanced)
- **Rectangular Patch-based**: Cut-paste approach within same image  
- **Probabilistic Generation**: 0.0~1.0 probability control (compatible with original DRAEM)
- **Adaptive Patch Sizing**: Automatic scaling based on input image dimensions
- **Configurable Parameters**:
  - Patch ratio: Landscape (>1.0), Portrait (<1.0), Square (1.0)
  - Patch width range: Pixel-based sizing (e.g., 20-80 pixels)
  - Severity range: 0 ~ user-defined max (e.g., 8.0)
  - Multi-patch support: 1-3 patches with identical properties
  - Anomaly probability: 0.5 (default), adjustable for normal/anomaly ratio

### 🧠 Severity Prediction Modes (5 Ablation Options)
1. `discriminative_only`: Baseline (2 channels) - Pure discriminative output
2. `with_original`: + Original image (5 channels) - Original context information  
3. `with_reconstruction`: + Reconstruction (5 channels) - Reconstruction quality
4. `with_error_map`: + Reconstruction error (5 channels) - Explicit error information
5. `multi_modal`: All information combined (11 channels) - Maximum information fusion

### 🚀 Implementation Features
- **Flexible Image Size**: Supports any square input (NxN) via AdaptiveAvgPool2d
- **Memory Management**: Systematic GPU memory optimization
- **CUDA Support**: Efficient GPU acceleration with memory cleanup
- **Modular Design**: Easy component swapping for ablation studies

## 📊 Usage Example (Updated for Latest Implementation)

```python
from anomalib.models.image.custom_draem import CustomDraem
from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule
from anomalib.engine import Engine

# Initialize model with DRAEM backbone integration
model = CustomDraem(
    # Severity Sub-Network settings
    severity_input_mode="discriminative_only",  # 5 options available
    
    # Synthetic Fault Generation (enhanced)
    anomaly_probability=0.5,          # 50% anomaly generation rate
    patch_width_range=(32, 64),       # Patch size in pixels
    patch_ratio_range=(0.1, 0.5),     # Patch aspect ratios
    severity_max=8.0,                 # Maximum severity value
    patch_count=1,                    # Single patch (recommended)
    
    # Loss configuration
    reconstruction_weight=1.0,
    segmentation_weight=1.0,
    severity_weight=0.5,
    
    # Adaptive loss (advanced feature)
    use_adaptive_loss=True,           # Uncertainty-based weighting
    warmup_epochs=5,                  # Reconstruction-focused warmup
    
    # DRAEM backbone options
    sspcab=False,                     # Optional attention mechanism
    
    # Optimizer settings
    optimizer="adam",
    learning_rate=1e-4,
)

# Setup data
datamodule = MultiDomainHDMAPDataModule(
    root="./datasets/HDMAP/1000_8bit_resize_NxN",  # Any square size supported
    source_domain="domain_A",
    target_domains="auto",
    train_batch_size=16,
    eval_batch_size=16,
)

# Training configuration
engine = Engine(
    accelerator="gpu",
    devices=1,
    max_epochs=30,
    check_val_every_n_epoch=1,
    enable_checkpointing=True,
)

# Full training pipeline
engine.fit(model=model, datamodule=datamodule)

# Evaluation
source_results = engine.test(model=model, dataloaders=datamodule.val_dataloader())
target_results = engine.test(model=model, dataloaders=datamodule.test_dataloader())

# Results contain standard anomaly detection metrics
# AUROC, F1-Score, etc. plus custom severity prediction accuracy
```

### 🔬 **Advanced Usage: Direct Comparison with Original DRAEM**

```python
# For research: compare Custom DRAEM vs Original DRAEM
from anomalib.models.image.draem import Draem

# 1. Original DRAEM (baseline)
original_draem = Draem()

# 2. Custom DRAEM (same backbone + custom features)
custom_draem = CustomDraem(
    severity_input_mode="discriminative_only",  # Fair comparison
    use_adaptive_loss=False,                    # Disable custom loss
    severity_weight=0.0,                        # Disable severity head
    # Now both models have identical 97.4M backbone
)

# 3. Full Custom DRAEM (all features enabled)  
full_custom_draem = CustomDraem(
    severity_input_mode="multi_modal",          # Maximum information
    use_adaptive_loss=True,                     # Adaptive loss enabled
    severity_weight=0.5,                        # Severity head enabled
)

# Train all three with identical settings for ablation study
```

## 🔬 Experimental Setup

### 🏆 **Direct Comparison Studies**
1. **DRAEM vs Custom DRAEM**: Identical 97.4M backbone comparison
2. **Ablation Study**: Custom component contributions
   - Backbone only (97.4M params)
   - + Severity Head (+118K params) 
   - + Adaptive Loss (dynamic weighting)
3. **Domain Transfer Analysis**: Source vs Target domain adaptation

### 🧪 **Ablation Studies**
- **Severity Input Modes**: 5 different input combinations (2, 5, 11 channels)
- **Loss Functions**: Fixed weights vs Adaptive uncertainty-based weighting
- **Patch Configurations**: Landscape vs Portrait vs Square
- **Multi-patch**: 1 vs 2 vs 3 patches
- **Image Size Flexibility**: Any square NxN input supported

### 📊 **Default Parameters**
- **Batch Size**: 16 (adjustable based on GPU memory)
- **Learning Rate**: 0.0001 (consistent with original DRAEM)
- **Loss Weights**: 1.0:1.0:0.5 (reconstruction:segmentation:severity)
- **Anomaly Probability**: 0.5 (50% normal/anomaly ratio)
- **Patch Width Range**: 20-80 pixels (relative to input size)

## 📁 File Structure (Updated)

```
custom_draem/
├── __init__.py                 # Main export
├── lightning_model.py          # Lightning wrapper (AnomalibModule) - Updated for DRAEM backbone
├── torch_model.py             # Core PyTorch model - DRAEM integration + Severity head  
├── loss.py                    # Multi-task loss function - Standard weighted loss
├── adaptive_loss.py           # Advanced adaptive loss with uncertainty weighting
├── synthetic_generator.py      # HDMAP synthetic fault generator - Enhanced probabilistic
└── README.md                  # This file (Updated)

tests/unit/models/image/custom_draem/
├── __init__.py
├── test_torch_model.py         # Core model testing
├── test_lightning_model.py     # Lightning integration testing  
├── test_synthetic_generator.py # Synthetic generation testing
└── test_draem_custom_draem_comparison.py  # 7-Phase comprehensive comparison

examples/hdmap/
└── multi_domain_hdmap_custom_draem_training.py  # Updated training script
```

## 🎯 Target Use Cases

1. **HDMAP Anomaly Detection**: Industrial health monitoring datasets with RGB support
2. **Fair Model Comparison**: Research requiring identical baseline comparisons  
3. **Domain Transfer Learning**: Multi-domain scenarios with varying characteristics
4. **Severity Assessment**: Quantitative fault severity prediction with 5 input modes
5. **Ablation Studies**: Systematic component contribution analysis
6. **Educational Research**: Model architecture understanding and comparison
7. **Flexible Input Processing**: Support for various square image sizes (NxN)

## 🏗️ **Model Specifications**

### 📊 **Architecture Details**
- **Total Parameters**: 97.5M (97.4M backbone + 118K severity head)
- **Input Support**: Any square RGB images (NxN dimensions)
- **Backbone**: Original DRAEM (Wide ResNet encoder + decoder + discriminator)
- **Custom Components**: Fault Severity Sub-Network with 5 input modes
- **Output**: Reconstruction + Segmentation + Severity prediction

### 🔧 **Technical Features**
- **Adaptive Pooling**: AdaptiveAvgPool2d ensures flexible input size support
- **Multi-task Learning**: 3-component loss function with configurable weights
- **Probabilistic Generation**: Compatible with original DRAEM's anomaly generation
- **SSPCAB Integration**: Optional attention mechanism for enhanced performance

## 📚 References

- Original DRAEM: [A discriminatively trained reconstruction embedding for surface anomaly detection](https://arxiv.org/abs/2108.07610)
- **Custom DRAEM Implementation**: Enhanced with severity prediction and DRAEM backbone integration