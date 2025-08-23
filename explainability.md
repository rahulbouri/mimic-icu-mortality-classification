# Clinical Mortality Prediction Model with Attention-Based Explainability

This repository contains a clinical transformer model for predicting patient mortality using temporal clinical event sequences. The model uses Clinical-Longformer with a simplified, effective architecture that captures both temporal and feature patterns through multi-head attention.

## Model Architecture

### **Simplified Design Philosophy**
- **Single Transformer**: One transformer encoder with multiple attention heads
- **Concatenated Input**: Event type + Clinical-Longformer embeddings + Relative time + Positional encoding
- **Multi-Head Attention**: Naturally captures temporal, feature, and interaction patterns
- **Attention Pooling**: Learns which sequence positions are most important

### **Why This Approach Works**
1. **Natural Separation**: Multi-head attention naturally learns different aspects of the data
2. **Efficient Processing**: Single transformer processes all information together
3. **Rich Representations**: Each attention head can specialize in different patterns
4. **Interpretable**: Attention weights directly show what the model focuses on

## Input Data Structure

Each timestep contains concatenated features:
```
[event_type_1, event_type_2, ..., event_type_5, 
 bert_embed_1, bert_embed_2, ..., bert_embed_768,
 time_embed_1, time_embed_2, ..., time_embed_16]
Total: 789 dimensions per timestep
```

- **Event Type**: One-hot encoded event categories (5 dimensions)
- **Clinical Text**: Clinical-Longformer embeddings (768 dimensions)  
- **Temporal Information**: Relative time to final event (16 dimensions)
- **Positional Encoding**: Sinusoidal encoding for sequence position

## Training

### Running the Training Script
```bash
python train.py
```

### Training Features
- **Temporal-Aware Loss**: Weights recent events more heavily
- **Attention Weight Saving**: Saves attention weights every 50 batches
- **Real-Time Visualization**: Generates attention heatmaps during training
- **Checkpointing**: Saves best model and latest checkpoint

### Output Directories
- `trained_models/clinical_LF/checkpoints/`: Model checkpoints
- `trained_models/clinical_LF/plots/`: Training visualizations
- `trained_models/clinical_LF/attention_weights/`: Saved attention weights
- `trained_models/clinical_LF/explainability/`: Post-training analysis

## Explainability Analysis

### **1. Understanding Attention Weights**

The model saves attention weights at multiple levels:

#### **A. Layer Attention Weights**
```python
# Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
layer_attention_weights = attention_weights['layer_attention_weights']
```
- **What it shows**: How each transformer layer attends to different sequence positions
- **Interpretation**: Early layers learn local patterns, later layers learn global relationships
- **Clinical insight**: Which time periods are connected to which other time periods

#### **B. Attention Pooling Weights**
```python
# Shape: (batch_size, 1, seq_len)
pooling_weights = attention_weights['pooling_weights']
```
- **What it shows**: Which sequence positions are most important for the final prediction
- **Interpretation**: Higher weights = more important positions
- **Clinical insight**: Which clinical events were most predictive of mortality

#### **C. Transformer Outputs**
```python
# Shape: (batch_size, seq_len, d_model)
transformer_output = attention_weights['transformer_output']
```
- **What it shows**: Learned representations for each timestep
- **Interpretation**: Higher activation = more important features
- **Clinical insight**: Which clinical features mattered most at each time point

### **2. Post-Training Analysis**

After training, run comprehensive explainability analysis:

```python
from train import comprehensive_explainability_analysis

# Load your trained model and test loader
comprehensive_explainability_analysis(model, test_loader, preprocessor)
```

This generates:
- `layer_attention_weights.png`: Attention patterns from each transformer layer
- `sequence_importance.png`: Which sequence positions were most important
- `feature_importance.png`: Which clinical features mattered most
- `query_vector_analysis.png`: Analysis of the learned query vector
- `captum_attributions.png`: Gradient-based feature attributions

### **3. Manual Attention Analysis**

#### **A. Analyze Which Time Periods Were Important**
```python
def analyze_critical_time_periods(attention_weights, threshold=0.1):
    """
    Identify which time periods were most critical for predictions
    """
    pooling_weights = attention_weights['pooling_weights']
    
    # Get position importance for first batch
    position_importance = pooling_weights[0, 0, :].cpu().numpy()
    
    critical_positions = []
    for pos, importance in enumerate(position_importance):
        if importance > threshold:
            critical_positions.append({
                'position': pos,
                'importance': importance,
                'interpretation': f"Events at position {pos} were {importance:.3f} important"
            })
    
    return critical_positions
```

#### **B. Analyze Feature Importance Across Time**
```python
def analyze_feature_importance_over_time(attention_weights):
    """
    Analyze how feature importance changes across time
    """
    transformer_output = attention_weights['transformer_output']
    
    # Average across batches
    avg_output = transformer_output.mean(dim=0)  # (seq_len, d_model)
    
    # Analyze each timestep
    timestep_analysis = []
    for timestep in range(avg_output.shape[0]):
        features = avg_output[timestep]
        
        # Find most important features at this timestep
        top_features = torch.topk(features, k=10)
        
        timestep_analysis.append({
            'timestep': timestep,
            'top_features': top_features.indices.cpu().numpy(),
            'top_importance': top_features.values.cpu().numpy()
        })
    
    return timestep_analysis
```

#### **C. Analyze Attention Patterns Between Layers**
```python
def analyze_layer_attention_patterns(attention_weights):
    """
    Analyze how attention patterns evolve across transformer layers
    """
    layer_weights = attention_weights['layer_attention_weights']
    
    layer_analysis = []
    for layer_idx, layer_attn in enumerate(layer_weights):
        # Average across batches and heads
        avg_attn = layer_attn.mean(dim=(0, 1))  # (seq_len, seq_len)
        
        # Analyze attention patterns
        layer_analysis.append({
            'layer': layer_idx + 1,
            'attention_matrix': avg_attn.cpu().numpy(),
            'global_attention': avg_attn.mean().item(),
            'local_attention': avg_attn.diagonal().mean().item()
        })
    
    return layer_analysis
```

### **4. Clinical Interpretation Examples**

#### **Example 1: Critical Time Window Identification**
```python
# Find which time periods were most important
critical_periods = analyze_critical_time_periods(attention_weights, threshold=0.15)

for period in critical_periods:
    print(f"Critical time period: Position {period['position']}")
    print(f"Importance: {period['importance']:.3f}")
    print(f"Clinical interpretation: Events at this time were highly predictive")
```

#### **Example 2: Feature Evolution Over Time**
```python
# Analyze how feature importance changes
feature_evolution = analyze_feature_importance_over_time(attention_weights)

for timestep in feature_evolution:
    print(f"\nTimestep {timestep['timestep']}:")
    print("Top features:", timestep['top_features'][:5])
    print("Importance:", timestep['top_importance'][:5])
```

#### **Example 3: Layer-Specific Patterns**
```python
# Analyze attention patterns across layers
layer_patterns = analyze_layer_attention_patterns(attention_weights)

for layer in layer_patterns:
    print(f"\nLayer {layer['layer']}:")
    print(f"Global attention: {layer['global_attention']:.3f}")
    print(f"Local attention: {layer['local_attention']:.3f}")
```

### **5. Captum Integration**

For advanced explainability, use Captum:

```python
from captum.attr import IntegratedGradients

def captum_analysis(model, sequences, static_features, attention_mask):
    """
    Use Captum for gradient-based attribution
    """
    explainer = IntegratedGradients(model)
    
    # Get attributions
    attributions = explainer.attribute(
        (sequences, static_features),
        target=1,  # Mortality prediction
        additional_forward_args=(attention_mask,)
    )
    
    return attributions
```

## Key Insights You Can Extract

### **1. Temporal Patterns**
- **Critical Time Windows**: "Events in the last 6 hours were most predictive"
- **Temporal Dependencies**: "Early events influenced later predictions"
- **Time Decay**: "Recent events had higher importance than older ones"

### **2. Feature Patterns**
- **Important Event Types**: "Lab results mattered more than medication changes"
- **Clinical Text Importance**: "Specific clinical descriptions were highly predictive"
- **Feature Interactions**: "Combination of lab + medication was critical"

### **3. Patient-Specific Patterns**
- **Risk Stratification**: "This patient type showed different temporal patterns"
- **Intervention Timing**: "Critical events happened earlier for high-risk patients"
- **Feature Sensitivity**: "This patient was more sensitive to certain event types"

## Clinical Decision Support

### **Risk Assessment**
- Identify which patients need closer monitoring
- Understand critical time windows for intervention
- Predict deterioration patterns

### **Quality Improvement**
- Learn from model insights to improve care protocols
- Identify gaps in clinical documentation
- Optimize monitoring schedules

### **Research Applications**
- Validate clinical hypotheses
- Discover new risk factors
- Understand disease progression patterns

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- Captum (for advanced explainability)
- scikit-learn
- matplotlib
- pandas
- numpy

## File Structure

```
mimic/
├── train.py                                    # Main training script
├── README.md                                   # This file
├── sample_train.csv                            # Sample training data
├── hf_cache/                                   # HuggingFace model cache
│   └── clinical_longformer/                    # Clinical-Longformer model
├── clinical_LF_precomp_emb/                    # Precomputed embeddings
├── trained_models/clinical_LF/                 # Training outputs
│   ├── checkpoints/                            # Model checkpoints
│   ├── plots/                                  # Training visualizations
│   ├── attention_weights/                      # Saved attention weights
│   └── explainability/                         # Post-training analysis
```

## Citation

If you use this model in your research, please cite:

```bibtex
@article{clinical_longformer,
  title={Clinical-Longformer: A Longformer-based Model for Clinical Named Entity Recognition},
  author={Yikuan Li and Yuan Luo and David Sontag},
  journal={arXiv preprint arXiv:2201.11838},
  year={2022}
}
```

## Summary

This architecture provides **effective explainability** through:

1. **Multi-Head Attention**: Naturally captures different aspects of clinical data
2. **Attention Weight Storage**: Persistent access to model decision patterns
3. **Layer-by-Layer Analysis**: Understanding of how patterns evolve
4. **Clinical Interpretability**: Direct mapping to clinical events and timing
5. **Comprehensive Analysis**: Multiple perspectives on model behavior

The key insight is that **simpler can be better** - a single transformer with proper attention analysis provides more interpretable results than complex multi-path architectures!
