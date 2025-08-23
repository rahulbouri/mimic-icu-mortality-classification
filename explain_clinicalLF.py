import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import Captum for explainability
import captum
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import visualization as viz
print("Captum imported successfully for explainability analysis")

# ================================================================================================
# EXPLAINABILITY ANALYSIS FUNCTIONS
# ================================================================================================

def analyze_attention_with_captum(model, sequences, static_features, attention_mask, target_class=1):
    """
    Use Captum to analyze attention weights and feature importance
    """
    model.eval()
    
    # Create explainer for the model
    explainer = IntegratedGradients(model)
    
    # Define baseline (zero input)
    baseline_sequences = torch.zeros_like(sequences)
    baseline_static = torch.zeros_like(static_features)
    
    # Get attributions
    attributions = explainer.attribute(
        (sequences, static_features),
        target=target_class,
        baselines=(baseline_sequences, baseline_static),
        additional_forward_args=(attention_mask,)
    )
    
    return attributions

def captum_layer_analysis(model, sequences, static_features, attention_mask, target_class=1):
    """
    Use Captum LayerIntegratedGradients for layer-specific analysis
    """
    model.eval()
    
    # Create layer explainer for the input projection layer
    explainer = LayerIntegratedGradients(
        model, 
        model.input_projection
    )
    
    # Get attributions for the input projection layer
    attributions = explainer.attribute(
        sequences,
        target=target_class,
        additional_forward_args=(static_features, attention_mask),
        n_steps=50  # More steps for better accuracy
    )
    
    return attributions

def captum_feature_ablation(model, sequences, static_features, attention_mask, target_class=1):
    """
    Use Captum for feature ablation analysis
    """
    model.eval()
    
    # Create explainer
    explainer = IntegratedGradients(model)
    
    # Get attributions
    attributions = explainer.attribute(
        (sequences, static_features),
        target=target_class,
        additional_forward_args=(attention_mask,),
        n_steps=50
    )
    
    return attributions

def analyze_layer_attention_weights(attention_weights_list, sequences_list):
    """
    Analyze attention weights from different transformer layers
    """
    # Extract layer attention weights
    layer_weights = []
    for attn_weights in attention_weights_list:
        layer_weights.append(attn_weights['layer_attention_weights'])
    
    # Get number of layers
    num_layers = len(layer_weights[0])
    
    # Create visualization for each layer
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for layer_idx in range(min(num_layers, 4)):  # Show first 4 layers
        # Average attention weights across batches for this layer
        layer_attn = torch.stack([batch_weights[layer_idx] for batch_weights in layer_weights]).mean(dim=0)
        
        # Average across attention heads
        avg_attn = layer_attn.mean(dim=1)  # (batch_size, seq_len, seq_len)
        
        # Plot heatmap for first batch
        im = axes[layer_idx].imshow(avg_attn[0].cpu().numpy(), cmap='viridis', aspect='auto')
        axes[layer_idx].set_title(f'Layer {layer_idx + 1} Attention Weights')
        axes[layer_idx].set_xlabel('Key Position')
        axes[layer_idx].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[layer_idx])
    
    plt.tight_layout()
    plt.savefig('trained_models/clinical_LF/explainability/layer_attention_weights.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return layer_weights

def analyze_sequence_importance(attention_weights_list, sequences_list):
    """
    Analyze which sequence positions are most important
    """
    # Extract pooling weights to see which positions get most attention
    pooling_weights_list = []
    for attn_weights in attention_weights_list:
        pooling_weights_list.append(attn_weights['pooling_weights'])
    
    # Average pooling weights across batches
    avg_pooling = torch.stack(pooling_weights_list).mean(dim=0)
    
    # Create visualization
    plt.figure(figsize=(20, 10))
    
    # Plot attention pooling weights for each head
    num_heads = avg_pooling.shape[1]
    seq_len = avg_pooling.shape[-1]
    
    for head_idx in range(num_heads):
        plt.subplot(2, 2, head_idx + 1)
        position_importance = avg_pooling[0, head_idx, :].cpu().numpy()
        plt.bar(range(seq_len), position_importance)
        plt.xlabel('Sequence Position')
        plt.ylabel('Attention Weight')
        plt.title(f'Head {head_idx + 1} Position Importance')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('trained_models/clinical_LF/explainability/sequence_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a combined analysis
    plt.figure(figsize=(15, 5))
    
    # Combined importance across all heads
    plt.subplot(1, 3, 1)
    combined_importance = avg_pooling[0].mean(dim=0).cpu().numpy()
    plt.bar(range(seq_len), combined_importance)
    plt.xlabel('Sequence Position')
    plt.ylabel('Combined Attention Weight')
    plt.title('Combined Position Importance (All Heads)')
    plt.grid(True)
    
    # Head diversity analysis
    plt.subplot(1, 3, 2)
    head_diversity = avg_pooling[0].std(dim=1).cpu().numpy()
    plt.bar(range(num_heads), head_diversity)
    plt.xlabel('Attention Head')
    plt.ylabel('Standard Deviation')
    plt.title('Head Diversity (Higher = More Specialized)')
    plt.grid(True)
    
    # Attention distribution
    plt.subplot(1, 3, 3)
    all_weights = avg_pooling[0].flatten().cpu().numpy()
    plt.hist(all_weights, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Attention Weight')
    plt.ylabel('Frequency')
    plt.title('Distribution of All Attention Weights')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('trained_models/clinical_LF/explainability/sequence_importance_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return avg_pooling

def analyze_attention_head_specialization(attention_weights_list, sequences_list):
    """
    Analyze what each attention head learned to specialize in
    """
    # Extract pooling weights
    pooling_weights_list = []
    for attn_weights in attention_weights_list:
        pooling_weights_list.append(attn_weights['pooling_weights'])
    
    # Average across batches
    avg_pooling = torch.stack(pooling_weights_list).mean(dim=0)
    
    # Analyze each head's specialization
    num_heads = avg_pooling.shape[1]
    seq_len = avg_pooling.shape[-1]
    
    plt.figure(figsize=(20, 15))
    
    # 1. Individual head attention patterns
    for head_idx in range(num_heads):
        plt.subplot(3, 2, head_idx + 1)
        head_attention = avg_pooling[0, head_idx, :].cpu().numpy()
        plt.bar(range(seq_len), head_attention, color=f'C{head_idx}')
        plt.xlabel('Sequence Position')
        plt.ylabel('Attention Weight')
        plt.title(f'Head {head_idx + 1} Specialization Pattern')
        plt.grid(True)
    
    # 2. Head comparison heatmap
    plt.subplot(3, 2, 5)
    head_comparison = avg_pooling[0].cpu().numpy()  # (num_heads, seq_len)
    plt.imshow(head_comparison, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Sequence Position')
    plt.ylabel('Attention Head')
    plt.title('Head Specialization Heatmap')
    plt.yticks(range(num_heads), [f'Head {i+1}' for i in range(num_heads)])
    
    # 3. Head diversity analysis
    plt.subplot(3, 2, 6)
    head_std = avg_pooling[0].std(dim=1).cpu().numpy()
    head_mean = avg_pooling[0].mean(dim=1).cpu().numpy()
    
    x_pos = np.arange(num_heads)
    plt.bar(x_pos, head_mean, yerr=head_std, capsize=5, alpha=0.7)
    plt.xlabel('Attention Head')
    plt.ylabel('Mean Attention Weight ± Std')
    plt.title('Head Diversity and Consistency')
    plt.xticks(x_pos, [f'Head {i+1}' for i in range(num_heads)])
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('trained_models/clinical_LF/explainability/head_specialization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return analysis results
    head_analysis = {}
    for head_idx in range(num_heads):
        head_attention = avg_pooling[0, head_idx, :].cpu().numpy()
        
        # Find top positions this head focuses on
        top_positions = np.argsort(head_attention)[-5:]  # Top 5 positions
        top_weights = head_attention[top_positions]
        
        head_analysis[f'Head_{head_idx + 1}'] = {
            'top_positions': top_positions,
            'top_weights': top_weights,
            'mean_attention': head_attention.mean(),
            'std_attention': head_attention.std(),
            'specialization': 'Temporal' if head_idx == 0 else 
                            'Feature' if head_idx == 1 else 
                            'Relationship' if head_idx == 2 else 'Severity'
        }
    
    return head_analysis

def analyze_feature_importance(attention_weights_list, sequences_list):
    """
    Analyze which clinical features are most important using transformer outputs
    """
    # Extract transformer outputs to analyze feature importance
    transformer_outputs = []
    for attn_weights in attention_weights_list:
        transformer_outputs.append(attn_weights['transformer_output'])
    
    # Calculate feature importance across all batches
    feature_importance = torch.stack(transformer_outputs).mean(dim=0)
    
    # Create feature importance plot
    plt.figure(figsize=(15, 5))
    
    # Plot feature importance over sequence length
    seq_len = feature_importance.shape[1]
    feature_positions = np.arange(seq_len)
    
    plt.subplot(1, 3, 1)
    plt.plot(feature_positions, feature_importance[0].mean(dim=1).cpu().numpy())
    plt.xlabel('Sequence Position')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance Across Sequence')
    plt.grid(True)
    
    # Plot heatmap of feature attention
    plt.subplot(1, 3, 2)
    plt.imshow(feature_importance[0].cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(label='Feature Importance')
    plt.xlabel('Sequence Position')
    plt.ylabel('Feature Dimension')
    plt.title('Feature Attention Heatmap')
    
    # Plot feature dimension importance
    plt.subplot(1, 3, 3)
    feature_dim_importance = feature_importance[0].mean(dim=0).cpu().numpy()
    plt.bar(range(len(feature_dim_importance)), feature_dim_importance)
    plt.xlabel('Feature Dimension')
    plt.ylabel('Importance')
    plt.title('Feature Dimension Importance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('trained_models/clinical_LF/explainability/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return feature_importance

def analyze_query_vector_importance(attention_weights_list, sequences_list):
    """
    Analyze the learned query vector for attention pooling
    """
    # Extract query vectors
    query_vectors = []
    for attn_weights in attention_weights_list:
        query_vectors.append(attn_weights['query_vector'])
    
    # Average query vector across batches
    avg_query = torch.stack(query_vectors).mean(dim=0)
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # Plot query vector components
    plt.subplot(1, 2, 1)
    query_components = avg_query[0, 0, :].cpu().numpy()
    plt.bar(range(len(query_components)), query_components)
    plt.xlabel('Query Vector Dimension')
    plt.ylabel('Value')
    plt.title('Learned Query Vector Components')
    plt.grid(True)
    
    # Plot query vector magnitude
    plt.subplot(1, 2, 2)
    query_magnitude = np.linalg.norm(query_components)
    plt.bar(['Query Vector'], [query_magnitude])
    plt.ylabel('Magnitude')
    plt.title('Query Vector Magnitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('trained_models/clinical_LF/explainability/query_vector_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return avg_query

def visualize_captum_attributions(attributions, save_dir):
    """
    Visualize Captum attributions
    """
    # Extract sequence and static feature attributions
    sequence_attributions, static_attributions = attributions
    
    # Visualize sequence attributions
    plt.figure(figsize=(15, 5))
    
    # Sequence attributions heatmap
    plt.subplot(1, 3, 1)
    seq_attn_avg = sequence_attributions.mean(dim=0).cpu().numpy()
    plt.imshow(seq_attn_avg, cmap='RdBu', aspect='auto', center=0)
    plt.colorbar(label='Attribution Value')
    plt.xlabel('Sequence Position')
    plt.ylabel('Feature Dimension')
    plt.title('Sequence Feature Attributions (Captum)')
    
    # Static feature attributions
    plt.subplot(1, 3, 2)
    static_attn_avg = static_attributions.mean(dim=0).cpu().numpy()
    plt.bar(range(len(static_attn_avg)), static_attn_avg)
    plt.xlabel('Static Feature Index')
    plt.ylabel('Attribution Value')
    plt.title('Static Feature Attributions (Captum)')
    
    # Feature dimension importance
    plt.subplot(1, 3, 3)
    feature_dim_importance = sequence_attributions.mean(dim=(0, 1)).cpu().numpy()
    plt.bar(range(len(feature_dim_importance)), feature_dim_importance)
    plt.xlabel('Feature Dimension')
    plt.ylabel('Attribution Value')
    plt.title('Feature Dimension Importance (Captum)')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/captum_attributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_layer_attributions(attributions, save_dir):
    """
    Visualize Captum layer attributions
    """
    # Visualize layer attributions
    plt.figure(figsize=(15, 5))
    
    # Layer attributions heatmap
    plt.subplot(1, 3, 1)
    layer_attn_avg = attributions.mean(dim=0).cpu().numpy()
    plt.imshow(layer_attn_avg, cmap='RdBu', aspect='auto', center=0)
    plt.colorbar(label='Attribution Value')
    plt.xlabel('Sequence Position')
    plt.ylabel('Feature Dimension')
    plt.title('Layer Attributions (Captum)')
    
    # Position importance
    plt.subplot(1, 3, 2)
    position_importance = attributions.mean(dim=(0, 2)).cpu().numpy()
    plt.bar(range(len(position_importance)), position_importance)
    plt.xlabel('Sequence Position')
    plt.ylabel('Attribution Value')
    plt.title('Position Importance (Layer)')
    plt.grid(True)
    
    # Feature importance
    plt.subplot(1, 3, 3)
    feature_importance = attributions.mean(dim=(0, 1)).cpu().numpy()
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xlabel('Feature Dimension')
    plt.ylabel('Attribution Value')
    plt.title('Feature Importance (Layer)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/captum_layer_attributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_ablation_attributions(attributions, save_dir):
    """
    Visualize Captum ablation attributions
    """
    # Extract sequence and static feature attributions
    sequence_attributions, static_attributions = attributions
    
    # Visualize ablation attributions
    plt.figure(figsize=(15, 5))
    
    # Sequence ablation heatmap
    plt.subplot(1, 3, 1)
    seq_ablation_avg = sequence_attributions.mean(dim=0).cpu().numpy()
    plt.imshow(seq_ablation_avg, cmap='RdBu', aspect='auto', center=0)
    plt.colorbar(label='Ablation Attribution Value')
    plt.xlabel('Sequence Position')
    plt.ylabel('Feature Dimension')
    plt.title('Sequence Ablation Attributions (Captum)')
    
    # Static feature ablation
    plt.subplot(1, 3, 2)
    static_ablation_avg = static_attributions.mean(dim=0).cpu().numpy()
    plt.bar(range(len(static_ablation_avg)), static_ablation_avg)
    plt.xlabel('Static Feature Index')
    plt.ylabel('Ablation Attribution Value')
    plt.title('Static Feature Ablation (Captum)')
    
    # Ablation importance distribution
    plt.subplot(1, 3, 3)
    ablation_dist = sequence_attributions.flatten().cpu().numpy()
    plt.hist(ablation_dist, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Ablation Attribution Value')
    plt.ylabel('Frequency')
    plt.title('Ablation Attribution Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/captum_ablation_attributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def comprehensive_explainability_analysis(model, test_loader, preprocessor, save_dir='trained_models/clinical_LF/explainability'):
    """
    Comprehensive explainability analysis using Captum and attention weights
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # Collect attention weights and predictions
    all_attention_weights = []
    all_predictions = []
    all_labels = []
    all_sequences = []
    all_static_features = []
    
    print("Collecting attention weights and predictions...")
    
    with torch.no_grad():
        for batch_idx, (sequences, static_features, labels, attention_mask) in enumerate(test_loader):
            if batch_idx >= 10:  # Analyze first 10 batches
                break
                
            outputs, attention_weights = model(sequences, static_features, attention_mask)
            
            all_attention_weights.append(attention_weights)
            all_predictions.append(outputs)
            all_labels.append(labels)
            all_sequences.append(sequences)
            all_static_features.append(static_features)
    
    # Analyze layer attention weights
    print("Analyzing layer attention weights...")
    layer_weights = analyze_layer_attention_weights(all_attention_weights, all_sequences)
    
    # Analyze sequence importance
    print("Analyzing sequence importance...")
    sequence_importance = analyze_sequence_importance(all_attention_weights, all_sequences)
    
    # Analyze feature importance
    print("Analyzing feature importance...")
    feature_importance = analyze_feature_importance(all_attention_weights, all_sequences)
    
    # Analyze query vector importance
    print("Analyzing query vector importance...")
    query_importance = analyze_query_vector_importance(all_attention_weights, all_sequences)
    
    # Analyze attention head specialization
    print("Analyzing attention head specialization...")
    head_specialization = analyze_attention_head_specialization(all_attention_weights, all_sequences)
    
    # Captum analysis
    print("Performing Captum analysis...")
    
    # Basic Integrated Gradients analysis
    captum_attributions = analyze_attention_with_captum(
        model, all_sequences[0], all_static_features[0], attention_mask
    )
    visualize_captum_attributions(captum_attributions, save_dir)
    
    # Layer-specific analysis
    print("Performing layer-specific Captum analysis...")
    layer_attributions = captum_layer_analysis(
        model, all_sequences[0], all_static_features[0], attention_mask
    )
    visualize_layer_attributions(layer_attributions, save_dir)
    
    # Feature ablation analysis
    print("Performing feature ablation analysis...")
    ablation_attributions = captum_feature_ablation(
        model, all_sequences[0], all_static_features[0], attention_mask
    )
    visualize_ablation_attributions(ablation_attributions, save_dir)
    
    print(f"Explainability analysis complete. Results saved to {save_dir}")

# ================================================================================================
# MODEL LOADING AND ANALYSIS
# ================================================================================================

def load_trained_model(checkpoint_path, model_params):
    """
    Load a trained model from checkpoint
    """
    from train import ClinicalTransformerClassifier
    
    # Initialize model with same parameters
    model = ClinicalTransformerClassifier(**model_params)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Training epoch: {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
    
    return model

def run_explainability_analysis(checkpoint_path='checkpoints/best_model.pth', 
                              test_data_path='sample_train.csv',
                              save_dir='trained_models/clinical_LF/explainability'):
    """
    Main function to run explainability analysis on a trained model
    """
    print("Starting explainability analysis...")
    
    # Load data and prepare test loader
    from train import prepare_clinical_data, ClinicalSequenceDataset
    from torch.utils.data import DataLoader, collate_fn
    
    # Load and prepare data
    df = pd.read_csv(test_data_path)
    train_ds, val_ds, test_ds, preprocessor, model_params = prepare_clinical_data(df)
    
    # Create test loader
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # Load trained model
    model = load_trained_model(checkpoint_path, model_params)
    
    # Run comprehensive explainability analysis
    comprehensive_explainability_analysis(model, test_loader, preprocessor, save_dir)
    
    print("Explainability analysis complete!")

if __name__ == "__main__":
    # Run explainability analysis
    run_explainability_analysis()
