# Severity-Aware DINOMALY Improvement

## 배경 및 동기

### 현재 상황
- **DINOMALY**: Binary anomaly detection만 제공 (정상/비정상)
- **실제 산업 요구**: 고장의 심각도 정보 필요
- **기존 연구**: DRAEM SevNet에서 severity 개념 시도했으나 성능 부족

### 핵심 아이디어
DINOMALY에 **Severity Prediction Head**를 추가하여 Multi-task Learning으로 anomaly detection + severity estimation 동시 수행

## 기술적 상세

### 1. Architecture 설계
```python
class SeverityDINOMALY(DinomalyModel):
    def __init__(self, ...):
        super().__init__(...)
        # Severity prediction head를 bottleneck 이후에 추가
        self.severity_head = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(), 
            nn.Linear(64, 1),
            nn.ReLU()  # 0~inf 범위 보장
        )
        self.severity_weight = 0.1  # Multi-task loss balancing
    
    def get_bottleneck_features(self, x):
        """Bottleneck features 추출 (severity prediction용)"""
        x = self.encoder.prepare_tokens(x)
        encoder_features = []
        
        for i, block in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = block(x)
            if i in self.target_layers:
                encoder_features.append(x)
        
        # Remove class token
        if self.remove_class_token:
            encoder_features = [e[:, 1 + self.encoder.num_register_tokens :, :] for e in encoder_features]
        
        # Fuse features and pass through bottleneck
        x = self._fuse_feature(encoder_features)
        for block in self.bottleneck:
            x = block(x)
            
        return x  # Bottleneck features
    
    def forward(self, batch, global_step=None, severity_target=None):
        en, de = self.get_encoder_decoder_outputs(batch)
        
        if self.training:
            # Reconstruction loss (기존 DINOMALY)
            recon_loss = self.loss_fn(encoder_features=en, decoder_features=de, global_step=global_step)
            
            if severity_target is not None:  # Augmented data with severity labels
                bottleneck_features = self.get_bottleneck_features(batch)
                # Global average pooling for severity prediction
                severity_features = bottleneck_features.mean(dim=1)  # (B, D)
                severity_pred = self.severity_head(severity_features)  # (B, 1)
                severity_loss = F.mse_loss(severity_pred.squeeze(), severity_target)
                
                total_loss = recon_loss + self.severity_weight * severity_loss
                return {
                    'loss': total_loss,
                    'recon_loss': recon_loss, 
                    'severity_loss': severity_loss,
                    'severity_pred': severity_pred.squeeze()
                }
            
            return {'loss': recon_loss, 'recon_loss': recon_loss}
        
        else:  # Inference
            anomaly_map, _ = self.calculate_anomaly_maps(en, de, out_size=batch.shape[2])
            bottleneck_features = self.get_bottleneck_features(batch)
            severity_features = bottleneck_features.mean(dim=1)
            severity_score = self.severity_head(severity_features)
            
            return InferenceBatch(
                anomaly_map=anomaly_map,
                pred_score=anomaly_map.flatten(1).max(dim=1)[0],  # Max anomaly score
                severity_score=severity_score.squeeze(),
                image=batch
            )
```

### 2. Severity Augmentation 전략
```python
def augment_with_severity(normal_image, max_severity=10.0):
    """정상 이미지에 severity에 따른 가로선 패턴 추가"""
    severity = torch.rand(1) * max_severity  # 0~max_severity 범위
    
    H, W = normal_image.shape[-2:]
    anomaly_pattern = torch.zeros_like(normal_image)
    
    # Severity에 비례한 가로선 생성
    num_lines = max(1, int(severity / 2))  # 심각도에 비례한 라인 수
    line_intensity = severity * 0.05       # 심각도에 비례한 밝기
    line_thickness = max(1, int(severity * 0.5))  # 심각도에 비례한 두께
    
    for _ in range(num_lines):
        y_pos = torch.randint(0, H, (1,))
        thickness_range = min(line_thickness, H - y_pos)
        
        # 가로선 추가 (가로선 = y축 고정, x축 전체)
        anomaly_pattern[:, :, y_pos:y_pos+thickness_range, :] = line_intensity
        
        # 부분적 끊어짐 효과 (realistic failure pattern)
        if torch.rand(1) > 0.7:  # 30% 확률로 끊어진 선
            break_start = torch.randint(0, W//2, (1,))
            break_end = torch.randint(W//2, W, (1,))
            anomaly_pattern[:, :, y_pos:y_pos+thickness_range, break_start:break_end] *= 0.3
    
    augmented_image = torch.clamp(normal_image + anomaly_pattern, 0, 1)
    return augmented_image, severity.item()

# Training data generation
def create_severity_training_batch(normal_batch):
    """정상 배치의 일부를 augmentation하여 severity training data 생성"""
    B = normal_batch.shape[0]
    aug_ratio = 0.3  # 30%만 augmentation
    num_aug = int(B * aug_ratio)
    
    if num_aug == 0:
        return normal_batch, None
    
    # 일부만 augmentation
    aug_indices = torch.randperm(B)[:num_aug]
    severity_targets = torch.zeros(B)
    
    for i, idx in enumerate(aug_indices):
        normal_batch[idx], severity = augment_with_severity(normal_batch[idx])
        severity_targets[idx] = severity
    
    # Augmented sample이 있는 경우만 severity target 반환
    return normal_batch, severity_targets
```

### 3. Training Strategy
```python
def train_severity_dinomaly():
    model = SeverityDINOMALY(...)
    
    for epoch in range(num_epochs):
        for batch_idx, normal_batch in enumerate(dataloader):
            # Severity augmentation
            batch, severity_targets = create_severity_training_batch(normal_batch)
            
            # Forward pass
            outputs = model(batch, global_step=global_step, severity_target=severity_targets)
            
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            if severity_targets is not None:
                logger.log({
                    'total_loss': outputs['loss'],
                    'recon_loss': outputs['recon_loss'],
                    'severity_loss': outputs['severity_loss'],
                    'severity_mae': F.l1_loss(outputs['severity_pred'], severity_targets[severity_targets > 0])
                })
```

## 장단점 분석

### 장점
1. **실용성**: 산업 현장에서 바로 활용 가능한 severity 정보 제공
2. **End-to-End Learning**: Reconstruction과 severity prediction 상호 보완
3. **연속적 Severity**: 0~inf 범위로 자연스러운 modeling
4. **Multi-task Regularization**: Severity task가 reconstruction 성능 향상에 기여 가능

### 단점
1. **제한적 독창성**: Multi-task learning 자체는 흔한 접근법
2. **Augmentation 의존성**: Severity label 품질이 성능에 직접적 영향  
3. **Ground Truth 부족**: 실제 severity 검증 어려움
4. **Hyperparameter Sensitivity**: Loss balancing weight 등 튜닝 필요

## 실험 설계

### Ablation Studies
1. **Severity Weight**: 0.01, 0.05, 0.1, 0.2, 0.5 비교
2. **Augmentation Ratio**: 10%, 30%, 50%, 70% 비교  
3. **Severity Head Architecture**: Different depth/width
4. **Feature Aggregation**: Average vs Max vs Attention pooling

### Evaluation Metrics
1. **Anomaly Detection**: AUROC, F1-score (기존 metric 유지)
2. **Severity Estimation**: MAE, MSE, Spearman correlation
3. **Consistency Check**: Anomaly score vs Severity score 상관관계

### 실제 Validation 방법
- **Synthetic Severity**: 다양한 강도의 인공 anomaly 생성하여 consistency 검증
- **Expert Annotation**: 실제 고장 케이스에 대한 전문가 severity 평가
- **Cross-validation**: 다른 도메인 데이터에서 severity prediction 일관성

## 논문 기여도 분석

### 강점
- **실용적 가치**: 산업 현장 직접 적용 가능
- **Comprehensive Solution**: Detection + Localization + Severity 통합 제공

### 약점 (예상 리뷰어 지적사항)
- **제한적 기술 혁신**: 단순히 regression head 추가한 수준
- **인위적 Ground Truth**: Augmentation으로 생성한 severity의 현실성 의문
- **유사 연구 존재**: 다른 도메인에서 multi-task anomaly detection 이미 연구됨

## 다음 단계 TODO
1. **[Week 1-2]** SeverityDINOMALY 모듈 구현
2. **[Week 2-3]** Severity augmentation 전략 실험 
3. **[Week 3-4]** Multi-task loss balancing 최적화
4. **[Week 4-5]** 다양한 severity evaluation metric 검증
5. **[Week 5-6]** 실제 데이터에서 expert annotation 수집

## 결론
Severity-Aware DINOMALY는 실용적 가치는 높지만, 논문 기여도 측면에서는 Directional Attention 대비 상대적으로 제한적. 
**추천**: Directional Attention을 main contribution으로, Severity-Aware를 secondary contribution으로 활용하는 것을 고려.