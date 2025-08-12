# Custom DRAEM for HDMAP Datasets

Custom DRAEM is an extended version of the original DRAEM (A discriminatively trained reconstruction embedding for surface anomaly detection) model, specifically designed for HDMAP (Health Data Map) datasets.

## 🎯 Key Features

### 🏗️ Architecture
- **Triple Sub-Network Design**:
  1. **Reconstructive Sub-Network**: 1-channel grayscale image reconstruction
  2. **Discriminative Sub-Network**: Anomaly segmentation and localization  
  3. **Fault Severity Sub-Network**: Continuous severity prediction

### 🔧 Synthetic Fault Generation
- **Rectangular Patch-based**: Cut-paste approach within same image
- **Configurable Parameters**:
  - Patch ratio: Landscape (>1.0), Portrait (<1.0), Square (1.0)
  - Patch size: 20-80 pixels (256x256 image basis)
  - Severity range: 0 ~ user-defined max (e.g., 10.0)
  - Multi-patch support: 1-3 patches with identical properties

### 🧠 Severity Prediction Modes
1. `discriminative_only`: Baseline (2 channels)
2. `with_original`: + Original image (3 channels)
3. `with_reconstruction`: + Reconstruction (4 channels)  
4. `with_error_map`: + Reconstruction error (4 channels)
5. `multi_modal`: All information combined (6 channels)

## 📊 Usage Example

```python
from anomalib.models.image import CustomDraem
from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule

# Initialize model
model = CustomDraem(
    severity_max=10.0,
    severity_input_mode="discriminative_only",
    patch_ratio_range=(2.0, 4.0),  # Landscape patches
    patch_size_range=(20, 80),     # 20-80 pixels
    patch_count=1                  # Single patch
)

# Setup data
datamodule = MultiDomainHDMAPDataModule(
    root="./datasets/HDMAP/1000_8bit_resize_256x256",
    source_domain="domain_A",
    target_domains="auto"
)

# Training (example)
from anomalib.engine import Engine
engine = Engine(task="segmentation", accelerator="gpu", devices=1)
engine.fit(model=model, datamodule=datamodule)
```

## 🔬 Experimental Setup

### Ablation Studies
- **Severity Input Modes**: 5 different input combinations
- **Patch Configurations**: Landscape vs Portrait vs Square
- **Multi-patch**: 1 vs 2 vs 3 patches
- **Loss Weights**: reconstruction:segmentation:severity ratios

### Recommended Parameters
- **Image Size**: 256x256 (preprocessed)
- **Batch Size**: 16 
- **Learning Rate**: 0.0001
- **Loss Weights**: 1.0:1.0:0.5 (adjustable)

## 📁 File Structure

```
custom_draem/
├── __init__.py                 # Main export
├── lightning_model.py          # Lightning wrapper (AnomalibModule)
├── torch_model.py             # Core PyTorch model (3 sub-networks)
├── loss.py                    # Multi-task loss function
├── synthetic_generator.py      # HDMAP synthetic fault generator
└── README.md                  # This file
```

## 🎯 Target Use Cases

1. **HDMAP Anomaly Detection**: Industrial health monitoring datasets
2. **Domain Transfer Learning**: Multi-domain scenarios with varying characteristics
3. **Severity Assessment**: Quantitative fault severity prediction
4. **Educational Research**: Ablation studies and model comparison

## 📚 References

- Original DRAEM: [A discriminatively trained reconstruction embedding for surface anomaly detection](https://arxiv.org/abs/2108.07610)