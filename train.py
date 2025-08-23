import os 

os.environ["HF_TOKEN"] = ""
print("Hugging Face API token configured!")

# Set up HuggingFace cache directory
cache_dir = '/Users/rahulbouri/Desktop/projects/mimic/hf_cache/clinical_longformer/'
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
print(f"Hugging Face cache configured at: {cache_dir}")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import copy

# ================================================================================================
# MODIFIED ATTENTION BLOCKS
# ================================================================================================

class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    """
    Like nn.TransformerEncoderLayer but forward returns (out, attn_weights).
    attn_weights is returned from MultiheadAttention with average_attn_weights=False
    so you typically get shape (batch, num_heads, L, S).
    """
    def __init__(self, d_model: int, nhead: int, *,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", layer_norm_eps: float = 1e-5,
                 batch_first: bool = True, norm_first: bool = False,
                 device=None, dtype=None):
        super().__init__(d_model=d_model,
                         nhead=nhead,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout,
                         activation=activation,
                         layer_norm_eps=layer_norm_eps,
                         batch_first=batch_first,
                         norm_first=norm_first,
                         device=device,
                         dtype=dtype)
        # ensure attribute exists for compatibility with some torch versions
        self.batch_first = batch_first

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        # get attn_output and per-head attn weights from MultiheadAttention
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,               # request weights
            average_attn_weights=False       # keep per-head weights
        )

        # same residual/ffn/norm as parent
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)

        return src, attn_weights


class TransformerEncoderWithAttn(nn.Module):
    """
    Simple wrapper that contains N TransformerEncoderLayerWithAttn and
    returns (output, layer_attns) where layer_attns is a list of attention tensors.
    """
    def __init__(self, encoder_layer: TransformerEncoderLayerWithAttn, num_layers: int, norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self,
                src: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        output = src
        layer_attns = []
        for layer in self.layers:
            output, attn = layer(output, mask, src_key_padding_mask)
            # attn is what MultiheadAttention returned; append as-is
            layer_attns.append(attn)
        if self.norm is not None:
            output = self.norm(output)
        return output, layer_attns

# Training-focused imports only
print("Training script loaded successfully!")

# ================================================================================================
# SECTION 1: DATA PREPROCESSING
# ================================================================================================

class ClinicalDataPreprocessor:
    """
    Handles all preprocessing steps for clinical data including:
    - One-hot encoding of event_type
    - ClinicalBERT embedding generation for event_text
    - Time encoding for relative_time_to_final_event
    - One-hot encoding of static features
    """
    
    def __init__(self, bert_model_name: str = "yikuan8/Clinical-Longformer", 
                 bert_embed_dim: int = 768, time_embed_dim: int = 16):
        """
        Initialize preprocessor with Clinical-Longformer model and embedding dimensions
        
        Args:
            bert_model_name: HuggingFace model name for Clinical-Longformer
            bert_embed_dim: Target dimension for embeddings (768 for Clinical-Longformer)
            time_embed_dim: Dimension for time encoding
        """
        self.bert_model_name = bert_model_name
        self.bert_embed_dim = bert_embed_dim
        self.time_embed_dim = time_embed_dim
        
        # Initialize Clinical-Longformer tokenizer and model with explicit cache directory
        cache_dir = os.environ.get('TRANSFORMERS_CACHE', './hf_cache/clinical_longformer/')
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name, cache_dir=cache_dir)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(
            bert_model_name, 
            cache_dir=cache_dir,
            output_hidden_states=True  # Ensure hidden states are returned
        )
        self.bert_model.eval()  # Set to evaluation mode
        print(f"Clinical-Longformer model and tokenizer loaded from cache: {cache_dir}")
        
        # No dimensionality reduction needed for Clinical-Longformer (768 dimensions)
        # self.bert_reducer = nn.Linear(768, bert_embed_dim)  # Removed
        
        # Initialize encoders (will be fitted during preprocessing)
        self.event_type_encoder = None
        self.static_encoders = {}
        self.time_scaler = StandardScaler()
        
        # Precomputed embeddings cache directory
        self.embeddings_cache_dir = "clinical_LF_precomp_emb/"
        os.makedirs(self.embeddings_cache_dir, exist_ok=True)
        
        # Store processed embeddings to avoid recomputation
        self.precomputed_embeddings = {}
        
        # Load existing precomputed embeddings if available
        self._load_precomputed_embeddings()
        
    def encode_event_type(self, event_types: pd.Series) -> np.ndarray:
        """
        One-hot encode event_type column using pre-fitted encoder
        
        Args:
            event_types: Series containing event type strings
            
        Returns:
            One-hot encoded array of shape (n_samples, n_event_types)
        """
        if self.event_type_encoder is None:
            raise ValueError("Event type encoder must be fitted before calling this method")
        
        # Transform event types using pre-fitted encoder
        encoded = self.event_type_encoder.transform(event_types.values.reshape(-1, 1))
        return encoded
    
    def encode_event_text_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate ClinicalBERT embeddings for event_text in batches with caching
        
        Args:
            texts: List of clinical text descriptions
            batch_size: Number of texts to process in each batch
            
        Returns:
            Array of embeddings with shape (n_texts, bert_embed_dim)
        """
        all_embeddings = []
        new_embeddings_count = 0
        cached_embeddings_count = 0
        
        # Clean and validate texts
        cleaned_texts = []
        for text in texts:
            if text is None or pd.isna(text):
                cleaned_texts.append("")  # Use empty string for missing text
            else:
                cleaned_texts.append(str(text))  # Convert to string
        
        # Process in batches to manage memory
        for i in range(0, len(cleaned_texts), batch_size):
            batch_texts = cleaned_texts[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                # Check if embedding is already cached
                text_hash = self._get_text_hash(text)
                
                if text_hash in self.precomputed_embeddings:
                    # Use cached embedding
                    embedding = self.precomputed_embeddings[text_hash]
                    cached_embeddings_count += 1
                else:
                    # Generate new embedding
                    inputs = self.tokenizer(
                        [text],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    
                    # Generate embedding without gradients (frozen Clinical-Longformer)
                    with torch.no_grad():
                        outputs = self.bert_model(**inputs)
                        # Clinical-Longformer returns hidden_states in outputs.hidden_states
                        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                            # Use the last layer's hidden states
                            cls_embedding = outputs.hidden_states[-1][0, 0, :]  # Shape: (1, 768)
                        else:
                            # Fallback to last_hidden_state if available
                            cls_embedding = outputs.last_hidden_state[0, 0, :]  # Shape: (1, 768)
                        
                        # No dimensionality reduction needed for Clinical-Longformer
                        embedding = cls_embedding.squeeze(0).numpy()  # Shape: (bert_embed_dim,)
                    
                    # Cache the new embedding
                    self.precomputed_embeddings[text_hash] = embedding
                    self._save_embedding(text_hash, embedding)
                    new_embeddings_count += 1
                
                batch_embeddings.append(embedding)
            
            all_embeddings.append(np.array(batch_embeddings))
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {min(i + batch_size, len(cleaned_texts))}/{len(cleaned_texts)} texts")
        
        embeddings = np.vstack(all_embeddings)
        
        if new_embeddings_count > 0:
            print(f"Generated {new_embeddings_count} new embeddings, used {cached_embeddings_count} cached embeddings")
        else:
            print(f"All {cached_embeddings_count} embeddings loaded from cache")
        
        return embeddings
    
    def encode_relative_time(self, times: pd.Series) -> np.ndarray:
        """
        Encode relative_time_to_final_event using pre-fitted time scaler
        
        Args:
            times: Series containing relative time values
            
        Returns:
            Time embeddings of shape (n_samples, 1) - single scaled value
        """
        if self.time_scaler is None:
            raise ValueError("Time scaler must be fitted before calling this method")
        
        # Normalize time values using pre-fitted scaler and return as single dimension
        times_scaled = self.time_scaler.transform(times.values.reshape(-1, 1))
        return times_scaled  # Shape: (n_samples, 1)
    
    def encode_static_features(self, df: pd.DataFrame, static_columns: List[str]) -> np.ndarray:
        """
        One-hot encode static features using pre-fitted encoders
        
        Args:
            df: DataFrame containing static features
            static_columns: List of column names to encode
            
        Returns:
            One-hot encoded static features
        """
        all_static_encoded = []
        
        for col in static_columns:
            if col not in self.static_encoders:
                raise ValueError(f"Encoder for column '{col}' must be fitted before calling this method")
            
            # Transform feature using pre-fitted encoder
            encoded = self.static_encoders[col].transform(df[col].values.reshape(-1, 1))
            all_static_encoded.append(encoded)
        
        if all_static_encoded:
            static_features = np.hstack(all_static_encoded)
            return static_features
        else:
            return np.empty((len(df), 0))
    
    def _get_text_hash(self, text: str) -> str:
        """
        Generate a hash for text to use as cache key
        
        Args:
            text: Input text string
            
        Returns:
            Hash string for caching
        """
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cache_file_path(self, text_hash: str) -> str:
        """
        Get the cache file path for a given text hash
        
        Args:
            text_hash: Hash of the text
            
        Returns:
            Full path to cache file
        """
        return os.path.join(self.embeddings_cache_dir, f"{text_hash}.npy")
    
    def _load_precomputed_embeddings(self):
        """
        Load existing precomputed embeddings from cache directory
        """
        if not os.path.exists(self.embeddings_cache_dir):
            return
            
        print(f"Loading precomputed embeddings from: {self.embeddings_cache_dir}")
        loaded_count = 0
        
        for filename in os.listdir(self.embeddings_cache_dir):
            if filename.endswith('.npy'):
                text_hash = filename[:-4]  # Remove .npy extension
                cache_path = os.path.join(self.embeddings_cache_dir, filename)
                try:
                    embedding = np.load(cache_path)
                    self.precomputed_embeddings[text_hash] = embedding
                    loaded_count += 1
                except Exception as e:
                    print(f"Warning: Could not load embedding from {filename}: {e}")
        
        print(f"Loaded {loaded_count} precomputed embeddings")
    
    def _save_embedding(self, text_hash: str, embedding: np.ndarray):
        """
        Save embedding to cache file
        
        Args:
            text_hash: Hash of the text
            embedding: BERT embedding array
        """
        cache_path = self._get_cache_file_path(text_hash)
        np.save(cache_path, embedding)

# ================================================================================================
# SECTION 2: CUSTOM DATASET CLASS
# ================================================================================================

class ClinicalSequenceDataset(Dataset):
    """
    Custom Dataset class for clinical sequential data
    Handles variable sequence lengths and proper batching
    """
    
    def __init__(self, sequences: Dict[str, List[np.ndarray]], 
                 static_features: np.ndarray, 
                 labels: np.ndarray,
                 max_seq_length: Optional[int] = None):
        """
        Initialize dataset
        
        Args:
            sequences: Dict with keys 'event_type', 'event_text', 'time' containing lists of arrays
            static_features: Array of static features for each patient
            labels: Binary labels for hospital mortality
            max_seq_length: Maximum sequence length for padding/truncation
        """
        self.sequences = sequences
        self.static_features = static_features
        self.labels = labels
        self.max_seq_length = max_seq_length
        
    def _get_max_length(self) -> int:
        """Get maximum sequence length across all patients"""
        return max(len(seq) for seq in self.sequences['event_type'])
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            transformer_input: Concatenated features for transformer (seq_len, feature_dim)
            static_features: Static features for this patient
            label: Binary mortality label
        """
        # Get sequences for this patient
        event_type_seq = self.sequences['event_type'][idx]
        event_text_seq = self.sequences['event_text'][idx]
        time_seq = self.sequences['time'][idx]
        
        # Use actual sequence length (no truncation)
        seq_len = len(event_type_seq)
        
        # Concatenate features for each time step
        transformer_input = []
        for i in range(seq_len):
            # Concatenate: event_type + event_text + time
            timestep_features = np.concatenate([
                event_type_seq[i],      # One-hot event type
                event_text_seq[i],      # BERT embeddings
                time_seq[i]            # Time encoding
            ])
            transformer_input.append(timestep_features)
        
        transformer_input = np.array(transformer_input)
        
        return (
            torch.FloatTensor(transformer_input),
            torch.FloatTensor(self.static_features[idx]),
            torch.FloatTensor([self.labels[idx]])
        )

# ================================================================================================
# SECTION 3: TRANSFORMER MODEL ARCHITECTURE
# ================================================================================================

class ClinicalTransformerClassifier(nn.Module):
    """
    Transformer-based classifier for clinical mortality prediction
    
    Architecture:
    1. Transformer encoder layers for sequential data
    2. Attention pooling to create fixed-size representation
    3. Concatenation with static features
    4. Fully connected classification head
    """
    
    def __init__(self, 
                 input_dim: int,           # 5 + 768 + 1 = 774 (event_type + BERT + time)
                 static_dim: int,          # Dimension of static features
                 d_model: int = 256,       # Transformer model dimension
                 nhead: int = 8,           # Number of attention heads
                 num_layers: int = 4,      # Number of transformer layers
                 dim_feedforward: int = 512, # Feedforward dimension
                 dropout: float = 0.1,     # Dropout rate
                 max_seq_length: int = None): # Maximum sequence length (None for dynamic)
        """
        Initialize the transformer classifier
        
        Args:
            input_dim: Input feature dimension per timestep
            static_dim: Static features dimension
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super(ClinicalTransformerClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.static_dim = static_dim
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Input projection layer (project input features to d_model)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding for sequences (will be created dynamically)
        self.positional_encoding = None
        self.max_positional_length = 0
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayerWithAttn(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Input shape: (batch_size, seq_len, d_model)
        )
        
        self.norm = nn.LayerNorm(d_model)

        self.transformer_encoder = TransformerEncoderWithAttn(
            encoder_layer, 
            num_layers=num_layers,
            norm=self.norm
        )
    
        # Using multiple heads for better explainability
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,  # Multiple heads for diverse attention patterns
            batch_first=True
        )
        
        # Learnable query vector for attention pooling
        self.query_vector = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Classification head
        combined_dim = d_model + static_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Create sinusoidal positional encodings dynamically
        
        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
            
        Returns:
            Positional encoding tensor of shape (max_len, d_model)
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def _ensure_positional_encoding(self, seq_len: int, device: torch.device):
        """
        Ensure positional encoding is available for the given sequence length
        
        Args:
            seq_len: Required sequence length
            device: Device to place tensor on
        """
        if self.positional_encoding is None or seq_len > self.max_positional_length:
            # Create new positional encoding with sufficient length
            max_len = max(seq_len, 1024)  # Add some buffer
            self.positional_encoding = self._create_positional_encoding(max_len, self.d_model)
            self.max_positional_length = max_len
            print(f"Created positional encoding for max length: {max_len}")
        
        # Return positional encoding for the required length
        return self.positional_encoding[:seq_len].to(device)
    
    def _extract_layer_attention_weights(self, input_tensor, attention_mask):
        """
        Extract attention weights from all transformer encoder layers
        """
        attention_weights = []
        x = input_tensor
        
        for layer in self.transformer_encoder.layers:
            # Get self-attention weights from the layer
            # We need to hook into the self-attention mechanism
            attn_output, attn_weights = layer.self_attn(
                x, x, x,
                attn_mask=None,
                key_padding_mask=attention_mask,
                need_weights=True
            )
            attention_weights.append(attn_weights)
            
            # Apply the rest of the layer
            x = layer.norm1(attn_output + x)
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(ff_output + x)
        
        return attention_weights
    
    def forward(self, sequence_input: torch.Tensor, 
                static_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            sequence_input: Sequential features (batch_size, seq_len, input_dim)
            static_features: Static features (batch_size, static_dim)
            attention_mask: Padding mask for sequences (batch_size, seq_len)
            
        Returns:
            Predicted probabilities (batch_size, 1)
        """
        batch_size, seq_len, _ = sequence_input.shape
        
        # Project input to model dimension
        projected_input = self.input_projection(sequence_input)  # (batch_size, seq_len, input_dim) ->(batch_size, seq_len, d_model)
        
        # Add positional encoding (ensure it's available for this sequence length)
        pos_encoding = self._ensure_positional_encoding(seq_len, sequence_input.device)
        pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        transformer_input = projected_input + pos_encoding
        transformer_input = self.dropout(transformer_input)
        
        # Pass through transformer encoder
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()  # True for pad -> OK for PyTorch
        else:
            src_key_padding_mask = None
        
        transformer_output, layer_attn_weights = self.transformer_encoder(
            transformer_input,
            mask=None,
            src_key_padding_mask=src_key_padding_mask
        )  # (batch_size, seq_len, d_model)
        
        # Attention pooling to get fixed-size representation
        query = self.query_vector.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        
        pooled_output, pooling_weights = self.attention_pooling(
            query,
            transformer_output,
            transformer_output,
            key_padding_mask=src_key_padding_mask
        )   # (batch_size, 1, d_model)
        
        pooled_output = pooled_output.squeeze(1)  # (batch_size, d_model)
        
        # Store attention weights for explainability analysis
        self.attention_weights = {
            'transformer_output': transformer_output,  # Full sequence representations
            'pooling_weights': pooling_weights,        # Attention pooling weights
            'query_vector': query,                     # Query vector used for pooling
            'layer_attention_weights': layer_attn_weights  # Attention from each layer
        }
        
        # Concatenate with static features
        combined_features = torch.cat([pooled_output, static_features], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        
        # Return both prediction and attention weights for explainability
        return output, self.attention_weights

# ================================================================================================
# SECTION 4: TRAINING UTILITIES
# ================================================================================================


def train_model(model, train_loader, val_loader, test_loader, 
                num_epochs=50, learning_rate=1e-4, device='cpu'):
    """
    Training function with temporal-aware loss and checkpointing
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Training history
    train_losses = []
    val_losses = []
    test_losses = []
    best_val_loss = float('inf')
    
    # Create output directory
    os.makedirs('trained_models/clinical_LF/checkpoints', exist_ok=True)
    os.makedirs('trained_models/clinical_LF/plots', exist_ok=True)
    
    print(f"Training on device: {device}")
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (sequences, static_features, labels, attention_mask) in enumerate(train_loader):
            sequences = sequences.to(device)
            static_features = static_features.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs, attention_weights = model(sequences, static_features, attention_mask)
            
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # if batch_idx % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        save_attention_weights(attention_weights, batch_idx, epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, static_features, labels, attention_mask in val_loader:
                sequences = sequences.to(device)
                static_features = static_features.to(device)
                labels = labels.to(device)
                attention_mask = attention_mask.to(device)
                
                outputs, _ = model(sequences, static_features, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Test phase (for monitoring)
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for sequences, static_features, labels, attention_mask in test_loader:
                sequences = sequences.to(device)
                static_features = static_features.to(device)
                labels = labels.to(device)
                attention_mask = attention_mask.to(device)
                
                outputs, _ = model(sequences, static_features, attention_mask)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Test Loss: {test_loss:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'test_loss': test_loss
            }, 'trained_models/clinical_LF/checkpoints/best_model.pth')
            print(f'  New best model saved! (Val Loss: {val_loss:.4f})')
        
        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'test_loss': test_loss
        }, 'trained_models/clinical_LF/checkpoints/latest_checkpoint.pth')
        
        # Plot training curves
        if (epoch + 1) % 10 == 0:
            plot_training_curves(train_losses, val_losses, test_losses, epoch + 1)
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    print("\nTraining complete! Use explain_clinicalLF.py for explainability analysis.")
    
    return model, train_losses, val_losses, test_losses

def plot_training_curves(train_losses, val_losses, test_losses, epoch):
    """
    Plot and save training curves
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.plot(test_losses, label='Test Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Curves (Epoch {epoch})')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Focus')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'trained_models/clinical_LF/plots/training_curves_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_attention_weights(attention_weights, batch_idx, epoch, save_dir='trained_models/clinical_LF/attention_weights'):
    import os
    import numpy as np
    
    os.makedirs(save_dir, exist_ok=True)
    
    def process_tensor_recursively(obj):
        """Recursively process tensors in nested structures"""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        elif isinstance(obj, list):
            return [process_tensor_recursively(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(process_tensor_recursively(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: process_tensor_recursively(value) for key, value in obj.items()}
        else:
            return obj
    
    # Process all attention weights recursively
    attention_data = process_tensor_recursively(attention_weights)
    
    # Save attention weights
    np.savez_compressed(
        f'{save_dir}/attention_epoch_{epoch}_batch_{batch_idx}.npz',
        **attention_data
    )


def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable sequence lengths
    
    Args:
        batch: List of samples from Dataset.__getitem__
        
    Returns:
        Batched tensors with proper attention masking
    """
    sequences, static_features, labels = zip(*batch)
    
    # Get actual sequence lengths
    seq_lengths = [seq.size(0) for seq in sequences]
    max_seq_len = max(seq_lengths)
    
    # Pad sequences to max length in batch
    padded_sequences = []
    for seq in sequences:
        if seq.size(0) < max_seq_len:
            # Pad with zeros
            padding_size = max_seq_len - seq.size(0)
            padding = torch.zeros(padding_size, seq.size(1), dtype=seq.dtype)
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    
    # Stack tensors
    sequences = torch.stack(padded_sequences)
    static_features = torch.stack(static_features)
    labels = torch.stack(labels)
    
    # Create attention mask (True for actual positions, False for padding)
    attention_mask = torch.zeros(len(batch), max_seq_len, dtype=torch.bool)
    for i, seq_len in enumerate(seq_lengths):
        attention_mask[i, :seq_len] = True
    
    return sequences, static_features, labels, attention_mask

# ================================================================================================
# SECTION 5: EXAMPLE USAGE AND DATA PREPARATION
# ================================================================================================

def prepare_clinical_data(df: pd.DataFrame, 
                         patient_id_col: str = 'patient_id',
                         test_size: float = 0.2,
                         max_seq_length: int = 128) -> Tuple:
    """
    Complete data preparation pipeline for clinical transformer model
    
    Args:
        df: Raw clinical dataframe
        patient_id_col: Column name for patient ID
        test_size: Proportion for test split
        max_seq_length: Maximum sequence length
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, preprocessor, model_params)
    """
    print("Starting clinical data preparation...")
    
    # Initialize preprocessor with time_embed_dim=1 for standard scaling
    preprocessor = ClinicalDataPreprocessor(time_embed_dim=1)
    
    # Define static feature columns
    static_columns = [
        'patient_gender', 'first_icu_careunit', 'last_icu_careunit',
        'admission_type', 'admission_location', 'discharge_location',
        'insurance', 'marital_status', 'patient_race', 
        'patient_age'
    ]
    
    # STEP 1: Fit all encoders globally on the entire dataset
    print("Fitting encoders globally on entire dataset...")
    
    # Fit event type encoder on all unique event types
    all_event_types = df['event_type'].unique()
    preprocessor.event_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    preprocessor.event_type_encoder.fit(all_event_types.reshape(-1, 1))
    print(f"Event types found globally: {all_event_types}")
    
    # Fit static feature encoders on all unique values
    for col in static_columns:
        if col != 'patient_age':
            # Handle missing values by filling with 'Unknown'
            df[col] = df[col].fillna('Unknown')
            
            # Fit encoder on all unique values for this column
            unique_values = df[col].unique()
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoder.fit(unique_values.reshape(-1, 1))
            preprocessor.static_encoders[col] = encoder
            
            print(f"Static feature '{col}' categories: {unique_values}")
        else:
            # Handle numerical age column
            df[col] = df[col].fillna(df[col].mean())
            encoder = StandardScaler()
            encoder.fit(df[col].values.reshape(-1, 1))
            preprocessor.static_encoders[col] = encoder
            print(f"Static feature '{col}' (numerical) - using StandardScaler")
    
    # Fit time scaler on all time values
    all_times = df['relative_time_to_final_event'].values
    preprocessor.time_scaler.fit(all_times.reshape(-1, 1))
    print("Time scaler fitted globally")
    
    # STEP 2: Process individual patients using fitted encoders
    print("Processing individual patients...")
    patient_groups = df.groupby(patient_id_col)
    
    # Prepare containers for processed data
    all_sequences = {'event_type': [], 'event_text': [], 'time': []}
    all_static_features = []
    all_labels = []
    patient_ids = []
    
    for patient_id, patient_data in patient_groups:
        if len(patient_data) == 0:
            continue
            
        # Sort by event time or any temporal column
        patient_data = patient_data.sort_values('relative_time_to_final_event', ascending=False)
        
        # No sequence length limitation - keep all events
        
        # Process sequential features using fitted encoders
        event_type_encoded = preprocessor.encode_event_type(patient_data['event_type'])
        event_text_encoded = preprocessor.encode_event_text_batch(patient_data['event_text'].tolist())
        time_encoded = preprocessor.encode_relative_time(patient_data['relative_time_to_final_event'])
        
        all_sequences['event_type'].append(event_type_encoded)
        all_sequences['event_text'].append(event_text_encoded)
        all_sequences['time'].append(time_encoded)
        
        # Process static features (use first row since they're patient-level)
        static_data = patient_data.iloc[0:1][static_columns]
        static_encoded = preprocessor.encode_static_features(static_data, static_columns)
        all_static_features.append(static_encoded[0])
        
        # Label (binary mortality prediction)
        label = patient_data.iloc[0]['patient_expired_in_hospital']
        all_labels.append(int(label) if pd.notna(label) else 0)
        
        patient_ids.append(patient_id)
        
        if len(all_labels) % 100 == 0:
            print(f"Processed {len(all_labels)} patients...")
    
    # Convert to numpy arrays
    all_static_features = np.array(all_static_features)
    all_labels = np.array(all_labels)
    
    print(f"Data preparation complete!")
    print(f"Total patients: {len(all_labels)}")
    print(f"Positive cases: {np.sum(all_labels)} ({np.mean(all_labels):.2%})")
    print(f"Average sequence length: {np.mean([len(seq) for seq in all_sequences['event_type']]):.1f}")
    
    # Split data
    indices = np.arange(len(all_labels))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=all_labels, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, stratify=all_labels[train_idx], random_state=42)
    
    # Create datasets
    def create_dataset(idx_list):
        return ClinicalSequenceDataset(
            sequences={k: [v[i] for i in idx_list] for k, v in all_sequences.items()},
            static_features=all_static_features[idx_list],
            labels=all_labels[idx_list],
            max_seq_length=None  # No sequence length constraint
        )
    
    train_dataset = create_dataset(train_idx)
    val_dataset = create_dataset(val_idx)
    test_dataset = create_dataset(test_idx)
    
    # Calculate model parameters
    # Get the number of features from the event type encoder
    if hasattr(preprocessor.event_type_encoder, 'n_features_out_'):
        event_type_dim = preprocessor.event_type_encoder.n_features_out_
    elif hasattr(preprocessor.event_type_encoder, '_n_features_outs'):
        event_type_dim = preprocessor.event_type_encoder._n_features_outs[0]
    else:
        # Fallback: calculate from the data
        event_type_dim = all_sequences['event_type'][0].shape[1]
    
    input_dim = event_type_dim + preprocessor.bert_embed_dim + preprocessor.time_embed_dim
    static_dim = all_static_features.shape[1]
    
    model_params = {
        'input_dim': input_dim,
        'static_dim': static_dim,
        'max_seq_length': None  # Dynamic sequence length
    }
    
    print(f"Model input dimensions: {input_dim} (sequential) + {static_dim} (static)")
    
    return train_dataset, val_dataset, test_dataset, preprocessor, model_params

# ================================================================================================
# EXAMPLE USAGE
# ================================================================================================

if __name__ == "__main__":
    # Example usage (assuming you have your dataframe loaded as 'df')
    
    # Load your data
    df = pd.read_csv('sample_train.csv')
    # df = df.sample(frac=0.1) ### take 10 % fraction of the data
    
    # Prepare data
    train_ds, val_ds, test_ds, preprocessor, model_params = prepare_clinical_data(df)

    print("MODEL PARAMS:")
    print(model_params)
    
    # Initialize model
    model = ClinicalTransformerClassifier(**model_params)
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # Train the model
    print("Starting model training...")
    trained_model, train_losses, val_losses, test_losses = train_model(
        model, train_loader, val_loader, test_loader,
        num_epochs=50, learning_rate=1e-4, device='auto'
    )
    
    print("Clinical Transformer Model Training Complete!")
    print("Key Components:")
    print("1. ClinicalDataPreprocessor - Handles all data preprocessing")
    print("2. ClinicalSequenceDataset - Custom dataset for variable sequences")  
    print("3. ClinicalTransformerClassifier - Transformer model with TFCAM")
    print("4. prepare_clinical_data - Complete data preparation pipeline")
    print("5. TemporalAwareLoss - Loss function with temporal weighting")
    print("6. Training function with checkpointing and visualization")