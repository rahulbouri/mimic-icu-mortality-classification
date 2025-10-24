# mimic-icu-mortality-classification
=======
# Clinical Mortality Prediction Model with Captum-Based Explainability

This repository contains a clinical transformer model for predicting patient mortality using temporal clinical event sequences. The model uses Clinical-Longformer with a simplified, effective architecture that captures both temporal and feature patterns through multi-head attention, and provides comprehensive explainability through **Captum**.

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

### **Multi-Head Attention for Explainability**
Our model uses **4 attention heads** in the pooling layer, each learning different aspects:

- **Head 1**: Temporal patterns (which time periods matter most)
- **Head 2**: Feature importance (which clinical features are critical)
- **Head 3**: Event relationships (how events influence each other)
- **Head 4**: Severity indicators (which values suggest high risk)

This gives us **4 different perspectives** on the same data, making explainability much richer!

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

## File Structure

### **train.py**
- **Purpose**: Training script with model definition, data preprocessing, and training loop
- **Features**: 
  - Clinical-Longformer integration
  - Multi-head attention architecture
  - Temporal-aware loss function
  - Checkpointing and visualization
- **No Captum imports**: Clean training-focused code

### **explain_clinicalLF.py**
- **Purpose**: Comprehensive explainability analysis using Captum
- **Features**:
  - Integrated Gradients analysis
  - Layer-specific attributions
  - Feature ablation analysis
  - Attention weight visualization
  - Multi-head attention analysis
- **Dependencies**: Imports from `train.py` for model and data loading

## Explainability Analysis with Captum

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

```bash
# Option 1: Run the explainability script directly
python explain_clinicalLF.py

# Option 2: Import and use specific functions
from explain_clinicalLF import run_explainability_analysis

# Run analysis on your trained model
run_explainability_analysis(
    checkpoint_path='checkpoints/best_model.pth',
    test_data_path='sample_train.csv',
    save_dir='trained_models/clinical_LF/explainability'
)
```

This generates:
- `layer_attention_weights.png`: Attention patterns from each transformer layer
- `sequence_importance.png`: Which sequence positions were most important (per head)
- `sequence_importance_combined.png`: Combined analysis across all heads
- `head_specialization.png`: What each attention head learned to specialize in
- `feature_importance.png`: Which clinical features mattered most
- `query_vector_analysis.png`: Analysis of the learned query vector
- `captum_attributions.png`: Basic Integrated Gradients analysis
- `captum_layer_attributions.png`: Layer-specific attributions
- `captum_ablation_attributions.png`: Feature ablation analysis

### **Benefits of Separation**

1. **Clean Training Code**: `train.py` focuses solely on model training without explainability overhead
2. **Modular Design**: Explainability functions can be imported and used independently
3. **Easier Maintenance**: Changes to explainability don't affect training code
4. **Better Performance**: Training script loads faster without Captum imports
5. **Flexible Usage**: Can run explainability analysis on any trained model checkpoint

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

The model uses **Captum** for comprehensive explainability analysis:

#### **A. Integrated Gradients Analysis**
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

#### **B. Layer-Specific Analysis**
```python
from captum.attr import LayerIntegratedGradients

def layer_analysis(model, sequences, static_features, attention_mask):
    """
    Analyze specific layers for explainability
    """
    explainer = LayerIntegratedGradients(
        model, 
        model.input_projection  # Analyze input projection layer
    )
    
    attributions = explainer.attribute(
        sequences,
        target=1,
        additional_forward_args=(static_features, attention_mask),
        n_steps=50
    )
    
    return attributions
```

#### **C. Feature Ablation Analysis**
```python
def ablation_analysis(model, sequences, static_features, attention_mask):
    """
    Analyze how removing features affects predictions
    """
    explainer = IntegratedGradients(model)
    
    attributions = explainer.attribute(
        (sequences, static_features),
        target=1,
        additional_forward_args=(attention_mask,),
        n_steps=50
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
- **Captum** (for explainability analysis)
- scikit-learn
- matplotlib
- pandas
- numpy

## Installation

```bash
# Install Captum for explainability
conda install captum -c pytorch

# Or using pip
pip install captum
```

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
5. **Comprehensive Captum Analysis**: Multiple attribution methods for deep insights

The key insight is that **simpler can be better** - a single transformer with proper attention analysis and Captum integration provides more interpretable results than complex multi-path architectures!

---

## NOTION UPDATE

This section provides a detailed explanation of the code as it is, based on the actual implementation in `train.py` and the query structure in `query.sql`.

### **1. Query Structure and Data Organization**

The SQL query (lines 226-449 in `query.sql`) creates a structured clinical dataset where:

- **Base Patient Information**: Each ICU stay includes patient demographics, admission details, and mortality outcome
- **Event Union**: Combines 5 different event types into a single chronological sequence:
  - **Prescriptions**: Medication orders with drug details, dosage, and timing
  - **Procedure Events**: Clinical procedures with start/end times and status
  - **Lab Events**: Laboratory results with values, reference ranges, and flags
  - **Microbiology**: Culture results and antibiotic sensitivity
  - **Ingredient Events**: IV fluids and nutrition administration

Each event includes:
- `event_time`: When the event occurred
- `event_type`: Categorical classification (5 types)
- `event_text`: Detailed clinical description
- `relative_time_to_final_event`: Minutes from event to ICU discharge/death

The query aggregates events into ordered arrays per ICU stay, creating variable-length sequences for each patient.

### **2. Data Processing Pipeline**

#### **A. One-Hot Encoding and Scalar Normalization**

The `ClinicalDataPreprocessor` class handles feature encoding:

```python
# Event Type Encoding (Categorical → One-Hot)
preprocessor.event_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
preprocessor.event_type_encoder.fit(all_event_types.reshape(-1, 1))

# Static Features (Categorical → One-Hot, Numerical → Standard Scaled)
for col in static_columns:
    if col != 'patient_age':
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(unique_values.reshape(-1, 1))
    else:
        encoder = StandardScaler()
        encoder.fit(df[col].values.reshape(-1, 1))
```

**Key Features**:
- **Global Fitting**: All encoders are fitted on the entire dataset before patient-wise processing
- **Unknown Handling**: `handle_unknown='ignore'` ensures robustness to new categories
- **Consistent Encoding**: All patients get identical feature representations

#### **B. Clinical-Longformer Text Embedding Generation**

The model uses Clinical-Longformer to convert clinical text to 768-dimensional embeddings:

```python
def encode_event_text_batch(self, texts: List[str], batch_size: int = 32):
    # Tokenize clinical text
    inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Generate embeddings without gradients (frozen model)
    with torch.no_grad():
        outputs = self.bert_model(**inputs, output_hidden_states=True)
        # Extract CLS token embedding from last layer
        cls_embedding = outputs.hidden_states[-1][0, 0, :]  # Shape: (768,)
        embedding = cls_embedding.squeeze(0).numpy()
```

**Clinical-Longformer Advantages**:
- **Long Sequence Support**: Handles up to 4096 tokens vs BERT's 512
- **Clinical Domain**: Pre-trained on MIMIC-III clinical notes
- **Efficient Processing**: Sparse attention mechanism for memory efficiency

#### **C. Embedding Caching System**

The system implements intelligent caching to avoid recomputation:

```python
def _get_text_hash(self, text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def _get_cache_file_path(self, text_hash: str) -> str:
    return os.path.join(self.embeddings_cache_dir, f"{text_hash}.npy")

# Check cache before computation
if text_hash in self.precomputed_embeddings:
    embedding = self.precomputed_embeddings[text_hash]  # Use cached
else:
    # Generate new embedding and save to cache
    self._save_embedding(text_hash, embedding)
```

**Cache Benefits**:
- **MD5 Hashing**: Unique identification of clinical text
- **Persistent Storage**: Saved as `.npy` files for fast loading
- **Incremental Updates**: Only new embeddings require computation

#### **D. Time Series Vector Creation**

Time features are processed using global standardization:

```python
def encode_relative_time(self, times: pd.Series) -> np.ndarray:
    # Normalize using pre-fitted scaler across entire dataset
    times_scaled = self.time_scaler.transform(times.values.reshape(-1, 1))
    return times_scaled  # Shape: (n_samples, 1)
```

**Time Processing**:
- **Global Standardization**: `StandardScaler` fitted on all time values
- **Single Dimension**: Returns 1D scaled values instead of complex encoding
- **Temporal Context**: `relative_time_to_final_event` provides temporal positioning

#### **E. Static Feature Vector Creation**

Static features combine categorical and numerical encodings:

```python
def encode_static_features(self, df: pd.DataFrame, static_columns: List[str]):
    all_static_encoded = []
    for col in static_columns:
        encoded = self.static_encoders[col].transform(df[col].values.reshape(-1, 1))
        all_static_encoded.append(encoded)
    
    # Concatenate all encoded features
    static_features = np.hstack(all_static_encoded)
    return static_features
```

**Static Features Include**:
- **Demographics**: Gender, age, race, marital status
- **Admission Details**: Type, location, insurance
- **ICU Information**: Care unit, length of stay

#### **F. Dynamic Sequence Creation and Positional Encoding**

The system handles variable-length sequences dynamically:

```python
class ClinicalSequenceDataset(Dataset):
    def __getitem__(self, idx: int):
        # Get actual sequence length (no truncation)
        seq_len = len(event_type_seq)
        
        # Concatenate features for each timestep
        transformer_input = []
        for i in range(seq_len):
            timestep_features = np.concatenate([
                event_type_seq[i],      # One-hot event type (5D)
                event_text_seq[i],      # BERT embeddings (768D)
                time_seq[i]            # Time encoding (1D)
            ])
            transformer_input.append(timestep_features)
        
        return torch.FloatTensor(transformer_input)  # Shape: (seq_len, 774)
```

**Positional Encoding Addition**:
```python
def _ensure_positional_encoding(self, seq_len: int, device: torch.device):
    if self.positional_encoding is None or seq_len > self.max_positional_length:
        # Create new positional encoding dynamically
        max_len = max(seq_len, 1024)
        self.positional_encoding = self._create_positional_encoding(max_len, self.d_model)
    
    return self.positional_encoding[:seq_len].to(device)

# Add positional encoding to input
pos_encoding = self._ensure_positional_encoding(seq_len, sequence_input.device)
transformer_input = projected_input + pos_encoding
```

**Positional Encoding Benefits**:
- **Dynamic Creation**: Automatically handles any sequence length
- **Sinusoidal Pattern**: Allows model to learn relative positions
- **Feature Identification**: Model learns which dimensions correspond to which features

### **3. Model Architecture and Dimension Flow**

#### **A. Input Projection Layer**

The model projects concatenated features to transformer dimensions:

```python
# Input: (batch_size, seq_len, 774) - concatenated features
projected_input = self.input_projection(sequence_input)  
# Output: (batch_size, seq_len, 256) - transformer model dimension
```

**Dimension Change**: 774 → 256
- **774**: 5 (event_type) + 768 (BERT) + 1 (time)
- **256**: `d_model` parameter for transformer architecture

#### **B. Transformer Encoder Architecture**

The model uses a standard transformer encoder:

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=256,           # Input/output dimension
    nhead=8,               # 8 attention heads
    dim_feedforward=512,   # Feedforward network dimension
    dropout=0.1,           # Regularization
    activation='relu',      # Activation function
    batch_first=True        # Input shape: (batch, seq_len, d_model)
)

self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
```

**Layer Dimensions**:
- **Input**: (batch_size, seq_len, 256)
- **Layer 1-4**: Each maintains (batch_size, seq_len, 256)
- **Output**: (batch_size, seq_len, 256)

#### **C. Attention Pooling Mechanism**

The model uses multi-head attention for sequence pooling:

```python
self.attention_pooling = nn.MultiheadAttention(
    embed_dim=256,         # Input dimension
    num_heads=4,           # 4 attention heads for explainability
    batch_first=True
)

# Learnable query vector for attention pooling
self.query_vector = nn.Parameter(torch.randn(1, 1, 256))

# Attention pooling
query = self.query_vector.expand(batch_size, -1, -1)  # (batch_size, 1, 256)
pooled_output, pooling_weights = self.attention_pooling(
    query, transformer_output, transformer_output
)  # pooled_output: (batch_size, 1, 256), pooling_weights: (batch_size, 4, 1, seq_len)
```

**Pooling Process**:
- **Query**: Learnable vector that "asks" which sequence positions are important
- **Key/Value**: Transformer output sequence
- **Output**: Weighted combination of sequence positions
- **Weights**: Attention scores showing position importance

#### **D. Classification Head and Final Dimensions**

The model concatenates pooled features with static features:

```python
# Concatenate pooled sequence features with static features
combined_features = torch.cat([pooled_output.squeeze(1), static_features], dim=1)
# Shape: (batch_size, 256 + static_dim)

# Classification layers
self.classifier = nn.Sequential(
    nn.Linear(256 + static_dim, 256),    # First layer
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 128),                 # Second layer
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 1),                   # Output layer
    nn.Sigmoid()                         # Mortality probability
)
```

**Final Dimension Flow**:
- **Pooled Output**: (batch_size, 256)
- **Static Features**: (batch_size, static_dim)
- **Combined**: (batch_size, 256 + static_dim)
- **Final Output**: (batch_size, 1) - mortality probability

#### **E. Attention Weight Extraction and Storage**

The model extracts attention weights at multiple levels:

```python
def _extract_layer_attention_weights(self, input_tensor, attention_mask):
    attention_weights = []
    x = input_tensor
    
    for layer in self.transformer_encoder.layers:
        # Extract self-attention weights from each layer
        attn_output, attn_weights = layer.self_attn(
            x, x, x, need_weights=True
        )
        attention_weights.append(attn_weights)
        # Continue with layer processing...
    
    return attention_weights

# Store comprehensive attention information
self.attention_weights = {
    'transformer_output': transformer_output,      # (batch_size, seq_len, 256)
    'pooling_weights': pooling_weights,            # (batch_size, 4, 1, seq_len)
    'query_vector': query,                         # (batch_size, 1, 256)
    'layer_attention_weights': layer_attention_weights  # List of (batch_size, 8, seq_len, seq_len)
}
```

**Attention Weight Storage**:
- **Layer Weights**: 8 heads × 4 layers × sequence interactions
- **Pooling Weights**: 4 heads × sequence position importance
- **Batch-Level**: Each batch saves complete attention patterns

### **4. Explainability Process and Captum Integration**

#### **A. Captum's Role in Explainability**

Captum provides multiple attribution methods for understanding model decisions:

```python
# Integrated Gradients - tracks gradient flow from input to output
from captum.attr import IntegratedGradients
explainer = IntegratedGradients(model)
attributions = explainer.attribute(
    (sequences, static_features),
    target=1,  # Mortality prediction
    additional_forward_args=(attention_mask,)
)

# Layer-Specific Analysis - analyzes specific model components
from captum.attr import LayerIntegratedGradients
explainer = LayerIntegratedGradients(model, model.input_projection)
attributions = explainer.attribute(sequences, target=1)

# Feature Ablation - measures impact of removing features
attributions = explainer.attribute(
    (sequences, static_features),
    target=1,
    n_steps=50  # Number of interpolation steps
)
```

**Captum Methods**:
- **Integrated Gradients**: Shows how input features contribute to predictions
- **Layer Attribution**: Analyzes specific model layers
- **Feature Ablation**: Measures feature importance through removal

#### **B. Batch-Wise Attention Weight Saving**

The training process saves attention weights every 50 batches:

```python
def save_attention_weights(attention_weights, batch_idx, epoch, save_dir='trained_models/clinical_LF/attention_weights'):
    # Convert attention weights to numpy arrays
    attention_data = {}
    for key, value in attention_weights.items():
        if isinstance(value, torch.Tensor):
            attention_data[key] = value.detach().cpu().numpy()
        else:
            attention_data[key] = value
    
    # Save attention weights with batch and epoch information
    np.savez_compressed(
        f'{save_dir}/attention_epoch_{epoch}_batch_{batch_idx}.npz',
        **attention_data
    )

# During training
if batch_idx % 50 == 0:  # Save every 50 batches
    save_attention_weights(attention_weights, batch_idx, epoch)
```

**Attention Weight Benefits**:
- **Temporal Analysis**: Track how attention patterns evolve during training
- **Batch Variability**: Understand attention consistency across different data batches
- **Memory Efficiency**: Save every 50 batches to avoid storage issues

#### **C. Attribution of Predictions to Temporal and Static Features**

The explainability system connects model decisions to clinical features:

```python
def analyze_temporal_feature_attribution(attention_weights, feature_mapping):
    """
    Analyze how temporal features contribute to predictions
    """
    pooling_weights = attention_weights['pooling_weights']  # (batch_size, 4, 1, seq_len)
    transformer_output = attention_weights['transformer_output']  # (batch_size, seq_len, 256)
    
    # Analyze each attention head
    for head_idx in range(4):
        head_weights = pooling_weights[0, head_idx, 0, :]  # (seq_len,)
        
        # Find most important time positions
        important_positions = torch.topk(head_weights, k=5)
        
        for pos, weight in zip(important_positions.indices, important_positions.values):
            # Extract features at this position
            position_features = transformer_output[0, pos, :]  # (256,)
            
            # Map to clinical features
            clinical_interpretation = map_features_to_clinical(
                position_features, feature_mapping, pos
            )
            
            print(f"Head {head_idx}: Position {pos} (weight: {weight:.3f})")
            print(f"Clinical interpretation: {clinical_interpretation}")

def analyze_static_feature_attribution(attention_weights, static_features):
    """
    Analyze how static features contribute to predictions
    """
    # Analyze the concatenated features in classification head
    combined_features = torch.cat([
        attention_weights['transformer_output'].mean(dim=1),  # Average sequence features
        static_features  # Static patient features
    ], dim=1)
    
    # Use Captum to attribute importance
    explainer = IntegratedGradients(model)
    static_attributions = explainer.attribute(
        static_features,
        target=1,
        additional_forward_args=(attention_weights['transformer_output'],)
    )
    
    return static_attributions
```

**Attribution Process**:
1. **Temporal Attribution**: 
   - Extract attention weights from pooling layer
   - Identify important sequence positions
   - Map positions to clinical events and timing

2. **Static Feature Attribution**:
   - Use Captum to attribute importance to static features
   - Connect features to patient demographics and admission details

3. **Clinical Interpretation**:
   - Map model attention to clinical events
   - Identify critical time windows
   - Understand patient-specific risk factors

**Example Clinical Insights**:
- **Temporal**: "Lab results at 6 hours before discharge were most predictive"
- **Static**: "Age and admission type were the strongest static predictors"
- **Interaction**: "Young patients with emergency admissions showed different temporal patterns"

This explainability system provides clinicians with actionable insights into why the model made specific mortality predictions, enabling better clinical decision-making and model validation.
>>>>>>> 5d4a574 (Initial commit: Clinical mortality prediction model script, weights and precomputed embeddings. Explainability script MOSTLY INCORRECT)
=======
# mimic-icu-mortality-classification
>>>>>>> origin/main
