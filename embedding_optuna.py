"""
Optuna Hyperparameter Search for RLF Embedding Model

This script performs hyperparameter optimization using Optuna for the RLF 
embedding model. It searches for configurations that maximize 2-class Recall
while maintaining False Alarm Rate above a specified threshold.

Features:
- Hyperparameter search for model architecture and training parameters
- Multi-objective optimization (Recall + FAR constraint)
- Per-trial model saving and metric logging
- Embedding visualization with time_diff information
- Integration with existing data pipeline from train_model.py

Usage:
    python embedding_optuna.py --n_trials 50 --min_far 0.02 --output_dir results/optuna
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from optuna.trial import Trial
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

from embedding_model import (
    RLFEmbeddingModel, EmbeddingModelConfig, create_embedding_model,
    compute_metrics, print_metrics, FRC_PER_SECOND
)


# =============================================================================
# Configuration Constants (all configurable via arguments)
# =============================================================================

DEFAULT_N_TRIALS = 50
DEFAULT_MIN_FAR = 0.02  # 2% minimum false alarm rate
DEFAULT_OUTPUT_DIR = "results/optuna"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PATIENCE = 10
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_RANDOM_SEED = 42

# Time series configuration
DEFAULT_TIME_STEPS = 35
DEFAULT_FEATURES_PER_STEP = 10
DEFAULT_INPUT_DIM = DEFAULT_TIME_STEPS * DEFAULT_FEATURES_PER_STEP  # 350

# Column names from train_model.py
TIMESTAMP_COLUMN = "frc_64us_10"
TIME_DIFF_COLUMN = "frc_diff"
LABEL_COLUMN = "rlf_reason"


@dataclass
class OptunaConfig:
    """Configuration for Optuna search."""
    n_trials: int = DEFAULT_N_TRIALS
    min_far: float = DEFAULT_MIN_FAR
    output_dir: str = DEFAULT_OUTPUT_DIR
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE
    train_ratio: float = DEFAULT_TRAIN_RATIO
    val_ratio: float = DEFAULT_VAL_RATIO
    test_ratio: float = DEFAULT_TEST_RATIO
    random_seed: int = DEFAULT_RANDOM_SEED
    input_dim: int = DEFAULT_INPUT_DIM
    frc_per_second: int = FRC_PER_SECOND
    time_steps: int = DEFAULT_TIME_STEPS
    features_per_step: int = DEFAULT_FEATURES_PER_STEP


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_and_preprocess_data(
    csv_paths: List[str],
    config: OptunaConfig,
    excluded_features: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load and preprocess data from CSV files.
    
    Args:
        csv_paths: List of paths to CSV files
        config: Optuna configuration
        excluded_features: Features to exclude from input
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, 
                  time_diffs_train, time_diffs_val, time_diffs_test, 
                  valid_mask_train, valid_mask_val, valid_mask_test, info)
    """
    if excluded_features is None:
        excluded_features = [
            "frc_64us_10", "frc_64us_1", "file_source_10", "file_source_1",
            "pci_1", "pci_2", "pci_3", "pci_4", "pci_5",
            "pci_6", "pci_7", "pci_8", "pci_9", "pci_10",
            "frc_diff"  # Will be used separately
        ]
    
    all_dfs = []
    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
            print(f"Loaded {csv_path}: {len(df)} rows")
    
    if not all_dfs:
        raise ValueError("No valid CSV files found")
    
    df = pd.concat(all_dfs, axis=0, ignore_index=True)
    print(f"Total samples: {len(df)}")
    
    # Process labels
    df[LABEL_COLUMN] = df[LABEL_COLUMN].replace([" ", np.nan], 3)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    
    # Extract time_diff
    if TIME_DIFF_COLUMN in df.columns:
        time_diffs = df[TIME_DIFF_COLUMN].values.astype(np.float32)
        valid_mask = ~np.isnan(time_diffs)
        time_diffs = np.nan_to_num(time_diffs, nan=0.0)
    else:
        time_diffs = np.zeros(len(df), dtype=np.float32)
        valid_mask = np.zeros(len(df), dtype=bool)
    
    # Get features
    feature_cols = [
        col for col in df.columns 
        if col not in excluded_features and col != LABEL_COLUMN
    ]
    
    # Remove _ho_ columns
    feature_cols = [col for col in feature_cols if "_ho_" not in col]
    
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y = df[LABEL_COLUMN].values.astype(np.int32)
    
    # Normalize features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X = (X - X_mean) / X_std
    
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    indices = np.arange(len(X))
    
    train_end = int(len(X) * config.train_ratio)
    val_end = int(len(X) * (config.train_ratio + config.val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    time_diffs_train = time_diffs[train_idx]
    time_diffs_val = time_diffs[val_idx]
    time_diffs_test = time_diffs[test_idx]
    
    valid_mask_train = valid_mask[train_idx].astype(np.float32)
    valid_mask_val = valid_mask[val_idx].astype(np.float32)
    valid_mask_test = valid_mask[test_idx].astype(np.float32)
    
    info = {
        "feature_cols": feature_cols,
        "input_dim": X.shape[1],
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "num_test": len(test_idx),
        "X_mean": X_mean,
        "X_std": X_std,
        "class_counts_train": np.bincount(y_train, minlength=4).tolist(),
        "class_counts_test": np.bincount(y_test, minlength=4).tolist()
    }
    
    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        time_diffs_train, time_diffs_val, time_diffs_test,
        valid_mask_train, valid_mask_val, valid_mask_test,
        info
    )


def create_synthetic_data(
    n_samples: int = 1000,
    config: OptunaConfig = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Create synthetic data for testing when real data is not available.
    
    Returns same format as load_and_preprocess_data.
    """
    if config is None:
        config = OptunaConfig()
    
    np.random.seed(config.random_seed)
    
    input_dim = config.input_dim
    
    # Generate features
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    
    # Generate labels with class imbalance
    # Class 0, 1, 2: RLF types (rare), Class 3: Normal (common)
    class_probs = [0.05, 0.03, 0.02, 0.90]  # 10% RLF total
    y = np.random.choice([0, 1, 2, 3], size=n_samples, p=class_probs).astype(np.int32)
    
    # Make RLF samples slightly different
    rlf_mask = y != 3
    X[rlf_mask] += 0.3
    
    # Generate time_diffs
    time_diffs = np.zeros(n_samples, dtype=np.float32)
    valid_mask = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(n_samples):
        if y[i] != 3:  # RLF sample
            time_diffs[i] = np.random.uniform(0, config.frc_per_second * 0.5)
            valid_mask[i] = 1.0
        elif np.random.rand() < 0.3:  # Some normal samples near RLF
            time_diffs[i] = np.random.uniform(
                config.frc_per_second * 0.5, 
                config.frc_per_second * 2
            )
            valid_mask[i] = 1.0
    
    # Split data
    n_train = int(n_samples * config.train_ratio)
    n_val = int(n_samples * config.val_ratio)
    
    X_train = X[:n_train]
    X_val = X[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    
    y_train = y[:n_train]
    y_val = y[n_train:n_train + n_val]
    y_test = y[n_train + n_val:]
    
    time_diffs_train = time_diffs[:n_train]
    time_diffs_val = time_diffs[n_train:n_train + n_val]
    time_diffs_test = time_diffs[n_train + n_val:]
    
    valid_mask_train = valid_mask[:n_train]
    valid_mask_val = valid_mask[n_train:n_train + n_val]
    valid_mask_test = valid_mask[n_train + n_val:]
    
    info = {
        "input_dim": input_dim,
        "num_train": len(X_train),
        "num_val": len(X_val),
        "num_test": len(X_test),
        "synthetic": True,
        "class_counts_train": np.bincount(y_train, minlength=4).tolist(),
        "class_counts_test": np.bincount(y_test, minlength=4).tolist()
    }
    
    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        time_diffs_train, time_diffs_val, time_diffs_test,
        valid_mask_train, valid_mask_val, valid_mask_test,
        info
    )


# =============================================================================
# Training Function
# =============================================================================

def train_model(
    model: RLFEmbeddingModel,
    optimizer: tf.keras.optimizers.Optimizer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    time_diffs_train: np.ndarray,
    valid_mask_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    time_diffs_val: np.ndarray,
    valid_mask_val: np.ndarray,
    config: OptunaConfig,
    trial: Optional[Trial] = None
) -> Dict[str, Any]:
    """
    Train the model with early stopping.
    
    Returns:
        Dictionary with training history and best metrics
    """
    batch_size = config.batch_size
    max_epochs = config.max_epochs
    patience = config.patience
    
    n_train = len(X_train)
    n_batches = (n_train + batch_size - 1) // batch_size
    
    best_val_loss = float('inf')
    best_val_recall = 0.0
    patience_counter = 0
    best_weights = None
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_recall': [],
        'val_far': []
    }
    
    for epoch in range(max_epochs):
        # Shuffle training data
        perm = np.random.permutation(n_train)
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]
        time_diffs_shuffled = time_diffs_train[perm]
        valid_mask_shuffled = valid_mask_train[perm]
        
        # Training loop
        epoch_losses = []
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            td_batch = time_diffs_shuffled[start_idx:end_idx]
            vm_batch = valid_mask_shuffled[start_idx:end_idx]
            
            loss_dict = model.train_step_custom(
                tf.constant(X_batch),
                tf.constant(y_batch),
                tf.constant(td_batch),
                tf.constant(vm_batch),
                optimizer
            )
            epoch_losses.append(loss_dict['total_loss'].numpy())
        
        train_loss = np.mean(epoch_losses)
        
        # Validation
        val_probs, val_embeddings = model(tf.constant(X_val), training=False)
        val_loss, _ = model.compute_loss(
            tf.constant(X_val),
            tf.constant(y_val),
            tf.constant(time_diffs_val),
            tf.constant(valid_mask_val),
            training=False
        )
        val_loss = val_loss.numpy()
        
        val_preds = np.argmax(val_probs.numpy(), axis=1)
        val_metrics = compute_metrics(y_val, val_preds)
        
        val_recall = val_metrics['recall_2class']
        val_far = val_metrics['false_alarm_rate']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_recall'].append(val_recall)
        history['val_far'].append(val_far)
        
        # Report to Optuna for pruning
        if trial is not None:
            trial.report(val_recall, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_recall = val_recall
            patience_counter = 0
            best_weights = [w.numpy() for w in model.trainable_weights]
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_recall={val_recall:.4f}, "
                  f"val_far={val_far:.4f}")
    
    # Restore best weights
    if best_weights is not None:
        for w, best_w in zip(model.trainable_weights, best_weights):
            w.assign(best_w)
    
    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'best_val_recall': best_val_recall,
        'epochs_trained': len(history['train_loss'])
    }


# =============================================================================
# Visualization
# =============================================================================

def create_embedding_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    time_diffs: np.ndarray,
    valid_mask: np.ndarray,
    output_path: str,
    title: str = "Embedding Visualization",
    frc_per_second: int = FRC_PER_SECOND
) -> None:
    """
    Create interactive embedding visualization with Plotly.
    
    Shows embeddings colored by class label, with lightness indicating
    time proximity to RLF, and hover info showing label and time diff.
    """
    from sklearn.decomposition import PCA
    
    # Reduce to 2D or 3D if needed
    embedding_dim = embeddings.shape[1]
    
    if embedding_dim > 3:
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)
    elif embedding_dim == 3:
        embeddings_3d = embeddings
    else:
        # Pad to 3D
        embeddings_3d = np.zeros((len(embeddings), 3))
        embeddings_3d[:, :embedding_dim] = embeddings
    
    # Convert time_diffs to seconds
    time_seconds = time_diffs / frc_per_second
    
    # Create color based on class
    class_names = ['RLF-0', 'RLF-1', 'RLF-2', 'Normal']
    colors = ['red', 'orange', 'yellow', 'blue']
    
    # Create hover text
    hover_text = []
    for i in range(len(labels)):
        label = labels[i]
        if valid_mask[i]:
            time_str = f"{time_seconds[i]:.3f}s"
        else:
            time_str = "N/A"
        hover_text.append(
            f"Label: {class_names[label]}<br>"
            f"Time to RLF: {time_str}<br>"
            f"Index: {i}"
        )
    
    # Create opacity based on time proximity (closer = more opaque)
    opacity = np.ones(len(labels)) * 0.3  # Default opacity
    for i in range(len(labels)):
        if valid_mask[i]:
            # Closer to RLF = higher opacity
            opacity[i] = 0.3 + 0.7 * np.exp(-time_seconds[i] / 1.0)
    
    # Create figure
    fig = go.Figure()
    
    for class_idx in range(4):
        mask = labels == class_idx
        if not np.any(mask):
            continue
        
        fig.add_trace(go.Scatter3d(
            x=embeddings_3d[mask, 0],
            y=embeddings_3d[mask, 1],
            z=embeddings_3d[mask, 2],
            mode='markers',
            name=class_names[class_idx],
            marker=dict(
                size=4,
                color=colors[class_idx],
                opacity=opacity[mask]
            ),
            text=[hover_text[i] for i in np.where(mask)[0]],
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        ),
        legend=dict(x=0, y=1)
    )
    
    # Save as HTML
    fig.write_html(output_path)
    print(f"Saved visualization to {output_path}")
    
    # Also save a 2D version
    output_path_2d = output_path.replace('.html', '_2d.html')
    
    fig_2d = px.scatter(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        color=[class_names[l] for l in labels],
        opacity=opacity,
        hover_data={
            'Time to RLF (s)': time_seconds,
            'Valid time': valid_mask.astype(bool)
        },
        title=title + " (2D)"
    )
    fig_2d.write_html(output_path_2d)


def save_confusion_matrix_plot(
    cm: np.ndarray,
    labels: List[str],
    output_path: str,
    title: str = "Confusion Matrix"
) -> None:
    """Save confusion matrix as HTML using Plotly."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )
    
    fig.write_html(output_path)


# =============================================================================
# Optuna Objective Function
# =============================================================================

def create_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    time_diffs_train: np.ndarray,
    valid_mask_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    time_diffs_val: np.ndarray,
    valid_mask_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    time_diffs_test: np.ndarray,
    valid_mask_test: np.ndarray,
    config: OptunaConfig
):
    """
    Create the Optuna objective function.
    
    Optimizes for 2-class Recall while constraining FAR >= min_far.
    """
    input_dim = X_train.shape[1]
    
    def objective(trial: Trial) -> float:
        # Hyperparameter suggestions
        embedding_dim = trial.suggest_int('embedding_dim', 4, 16)
        n_layers = trial.suggest_int('n_layers', 1, 4)
        
        hidden_layers = []
        for i in range(n_layers):
            units = trial.suggest_int(f'hidden_{i}', 32, 256, step=32)
            hidden_layers.append(units)
        
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        
        contrast_temperature = trial.suggest_float('contrast_temp', 0.05, 0.5)
        time_loss_weight = trial.suggest_float('time_loss_weight', 0.0, 0.5)
        contrast_loss_weight = trial.suggest_float('contrast_loss_weight', 0.0, 0.5)
        
        l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        
        # Class weights for handling imbalance
        rlf_weight = trial.suggest_float('rlf_weight', 1.0, 10.0)
        class_weights = [rlf_weight, rlf_weight, rlf_weight, 1.0]
        
        print(f"\nTrial {trial.number}: emb_dim={embedding_dim}, layers={hidden_layers}, "
              f"dropout={dropout_rate:.2f}, lr={learning_rate:.5f}")
        
        # Create model
        model, optimizer = create_embedding_model(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            contrast_temperature=contrast_temperature,
            time_loss_weight=time_loss_weight,
            contrast_loss_weight=contrast_loss_weight,
            class_weights=class_weights,
            use_batch_norm=use_batch_norm,
            l2_regularization=l2_reg
        )
        
        # Train model
        train_result = train_model(
            model, optimizer,
            X_train, y_train, time_diffs_train, valid_mask_train,
            X_val, y_val, time_diffs_val, valid_mask_val,
            config, trial
        )
        
        # Evaluate on test set
        test_probs, test_embeddings = model(tf.constant(X_test), training=False)
        test_preds = np.argmax(test_probs.numpy(), axis=1)
        test_metrics = compute_metrics(y_test, test_preds)
        
        recall = test_metrics['recall_2class']
        far = test_metrics['false_alarm_rate']
        
        print(f"  Test: Recall={recall:.4f}, FAR={far:.4f}, "
              f"Precision={test_metrics['precision_2class']:.4f}")
        
        # Save trial results
        trial_dir = os.path.join(config.output_dir, f"trial_{trial.number:04d}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Save model
        model.save_weights(os.path.join(trial_dir, "model_weights"))
        
        # Save metrics
        metrics_dict = {
            'trial_number': trial.number,
            'hyperparameters': trial.params,
            'test_metrics': {
                'recall_2class': float(recall),
                'precision_2class': float(test_metrics['precision_2class']),
                'false_alarm_rate': float(far),
                'f1_2class': float(test_metrics['f1_2class']),
                'accuracy_4class': float(test_metrics['accuracy_4class']),
                'confusion_matrix_2class': test_metrics['confusion_matrix_2class'].tolist(),
                'confusion_matrix_4class': test_metrics['confusion_matrix_4class'].tolist()
            },
            'training': {
                'epochs_trained': train_result['epochs_trained'],
                'best_val_loss': float(train_result['best_val_loss']),
                'best_val_recall': float(train_result['best_val_recall'])
            }
        }
        
        with open(os.path.join(trial_dir, "metrics.json"), 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Save embedding visualization
        create_embedding_visualization(
            test_embeddings.numpy(),
            y_test,
            time_diffs_test,
            valid_mask_test,
            os.path.join(trial_dir, "embedding_viz.html"),
            title=f"Trial {trial.number} Embeddings",
            frc_per_second=config.frc_per_second
        )
        
        # Save confusion matrices
        save_confusion_matrix_plot(
            test_metrics['confusion_matrix_2class'],
            ['Normal', 'RLF'],
            os.path.join(trial_dir, "cm_2class.html"),
            f"Trial {trial.number} - 2-Class Confusion Matrix"
        )
        
        save_confusion_matrix_plot(
            test_metrics['confusion_matrix_4class'],
            ['RLF-0', 'RLF-1', 'RLF-2', 'Normal'],
            os.path.join(trial_dir, "cm_4class.html"),
            f"Trial {trial.number} - 4-Class Confusion Matrix"
        )
        
        # Apply FAR constraint
        # If FAR is below threshold, penalize the objective
        if far < config.min_far:
            # Return negative value to indicate bad trial
            print(f"  Trial penalized: FAR {far:.4f} < min_far {config.min_far}")
            return recall - (config.min_far - far) * 10  # Penalty
        
        return recall
    
    return objective


# =============================================================================
# Main Entry Point
# =============================================================================

def run_optuna_search(
    csv_paths: Optional[List[str]] = None,
    config: Optional[OptunaConfig] = None,
    use_synthetic: bool = False
) -> Tuple[optuna.Study, Dict[str, Any]]:
    """
    Run Optuna hyperparameter search.
    
    Args:
        csv_paths: List of paths to CSV data files
        config: Optuna configuration
        use_synthetic: If True, use synthetic data (for testing)
        
    Returns:
        Tuple of (study, best_metrics)
    """
    if config is None:
        config = OptunaConfig()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load data
    if use_synthetic or csv_paths is None:
        print("Using synthetic data for demo...")
        data = create_synthetic_data(n_samples=2000, config=config)
    else:
        print("Loading real data...")
        data = load_and_preprocess_data(csv_paths, config)
    
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     time_diffs_train, time_diffs_val, time_diffs_test,
     valid_mask_train, valid_mask_val, valid_mask_test,
     info) = data
    
    # Update input_dim from data
    config.input_dim = X_train.shape[1]
    
    # Save data info
    with open(os.path.join(config.output_dir, "data_info.json"), 'w') as f:
        json.dump({
            k: v.tolist() if isinstance(v, np.ndarray) else v 
            for k, v in info.items()
        }, f, indent=2)
    
    # Create objective function
    objective = create_objective(
        X_train, y_train, time_diffs_train, valid_mask_train,
        X_val, y_val, time_diffs_val, valid_mask_val,
        X_test, y_test, time_diffs_test, valid_mask_test,
        config
    )
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        study_name='rlf_embedding_search',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    # Run optimization
    print(f"\nStarting Optuna search with {config.n_trials} trials...")
    print(f"Objective: Maximize 2-class Recall with FAR >= {config.min_far}")
    print("=" * 60)
    
    study.optimize(objective, n_trials=config.n_trials, show_progress_bar=True)
    
    # Print results
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value (Recall): {study.best_trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save study summary
    summary = {
        'best_trial': study.best_trial.number,
        'best_value': study.best_trial.value,
        'best_params': study.best_trial.params,
        'n_trials': len(study.trials),
        'config': asdict(config)
    }
    
    with open(os.path.join(config.output_dir, "study_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create optimization history plot
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(os.path.join(config.output_dir, "optimization_history.html"))
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(os.path.join(config.output_dir, "param_importances.html"))
    
    return study, summary


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Optuna Hyperparameter Search for RLF Embedding Model'
    )
    
    parser.add_argument(
        '--csv_paths', 
        nargs='+', 
        default=None,
        help='Paths to CSV data files'
    )
    parser.add_argument(
        '--n_trials', 
        type=int, 
        default=DEFAULT_N_TRIALS,
        help=f'Number of Optuna trials (default: {DEFAULT_N_TRIALS})'
    )
    parser.add_argument(
        '--min_far', 
        type=float, 
        default=DEFAULT_MIN_FAR,
        help=f'Minimum False Alarm Rate constraint (default: {DEFAULT_MIN_FAR})'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=DEFAULT_BATCH_SIZE,
        help=f'Training batch size (default: {DEFAULT_BATCH_SIZE})'
    )
    parser.add_argument(
        '--max_epochs', 
        type=int, 
        default=DEFAULT_MAX_EPOCHS,
        help=f'Maximum training epochs (default: {DEFAULT_MAX_EPOCHS})'
    )
    parser.add_argument(
        '--patience', 
        type=int, 
        default=DEFAULT_PATIENCE,
        help=f'Early stopping patience (default: {DEFAULT_PATIENCE})'
    )
    parser.add_argument(
        '--use_synthetic', 
        action='store_true',
        help='Use synthetic data for testing'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=DEFAULT_RANDOM_SEED,
        help=f'Random seed (default: {DEFAULT_RANDOM_SEED})'
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Create config
    config = OptunaConfig(
        n_trials=args.n_trials,
        min_far=args.min_far,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        random_seed=args.seed
    )
    
    # Run search
    study, summary = run_optuna_search(
        csv_paths=args.csv_paths,
        config=config,
        use_synthetic=args.use_synthetic or args.csv_paths is None
    )
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
