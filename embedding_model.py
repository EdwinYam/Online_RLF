"""
TensorFlow MLP Embedding Model for RLF Prediction

This module implements a Multi-Layer Perceptron (MLP) based embedding model
for Radio Link Failure (RLF) prediction. Features:

- Input: Flattened time series data (N x T x C -> N x (T*C), default 35x10=350)
- Multiple loss functions:
  - L_cls: 4-class weighted cross-entropy (+ 2-class metrics)
  - L_contrast: Supervised contrastive loss
  - L_time: Time-aware loss using frc_diff information
- Configurable architecture (depth, width, dropout, batch normalization)
- Embedding layer output for KNN integration

Time unit: 15625 frc = 1 second
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import os


# =============================================================================
# Configuration Constants
# =============================================================================

FRC_PER_SECOND = 15625
DEFAULT_INPUT_DIM = 350  # 35 time steps x 10 features
DEFAULT_EMBEDDING_DIM = 16
DEFAULT_NUM_CLASSES = 4
DEFAULT_HIDDEN_LAYERS = [128, 64]
DEFAULT_DROPOUT_RATE = 0.3
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_CONTRAST_TEMPERATURE = 0.1
DEFAULT_TIME_LOSS_WEIGHT = 0.1
DEFAULT_CONTRAST_LOSS_WEIGHT = 0.1
DEFAULT_CLS_LOSS_WEIGHT = 1.0


@dataclass
class EmbeddingModelConfig:
    """Configuration for the Embedding Model."""
    input_dim: int = DEFAULT_INPUT_DIM
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    num_classes: int = DEFAULT_NUM_CLASSES
    hidden_layers: List[int] = None
    dropout_rate: float = DEFAULT_DROPOUT_RATE
    use_batch_norm: bool = True
    learning_rate: float = DEFAULT_LEARNING_RATE
    contrast_temperature: float = DEFAULT_CONTRAST_TEMPERATURE
    time_loss_weight: float = DEFAULT_TIME_LOSS_WEIGHT
    contrast_loss_weight: float = DEFAULT_CONTRAST_LOSS_WEIGHT
    cls_loss_weight: float = DEFAULT_CLS_LOSS_WEIGHT
    class_weights: Optional[List[float]] = None  # Weights for 4 classes
    frc_per_second: int = FRC_PER_SECOND
    l2_regularization: float = 0.0001
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64]
        if self.class_weights is None:
            # Default: higher weight for RLF classes (0, 1, 2)
            self.class_weights = [3.0, 3.0, 3.0, 1.0]


# =============================================================================
# Custom Layers
# =============================================================================

class EmbeddingBlock(tf.keras.layers.Layer):
    """
    MLP block that produces embeddings.
    """
    
    def __init__(
        self,
        hidden_units: List[int],
        embedding_dim: int,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        l2_reg: float = 0.0001,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.l2_reg = l2_reg
        
        # Build layers
        self.dense_layers = []
        self.bn_layers = []
        self.dropout_layers = []
        
        regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
        
        for units in hidden_units:
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=regularizer
                )
            )
            if use_batch_norm:
                self.bn_layers.append(tf.keras.layers.BatchNormalization())
            self.dropout_layers.append(tf.keras.layers.Dropout(dropout_rate))
        
        # Embedding layer (no activation for contrastive learning)
        self.embedding_layer = tf.keras.layers.Dense(
            embedding_dim,
            activation=None,
            kernel_regularizer=regularizer,
            name='embedding'
        )
        
    def call(self, inputs, training=False):
        x = inputs
        
        for i, dense in enumerate(self.dense_layers):
            x = dense(x)
            if self.use_batch_norm:
                x = self.bn_layers[i](x, training=training)
            x = self.dropout_layers[i](x, training=training)
        
        embedding = self.embedding_layer(x)
        return embedding
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_units': self.hidden_units,
            'embedding_dim': self.embedding_dim,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'l2_reg': self.l2_reg
        })
        return config


# =============================================================================
# Loss Functions
# =============================================================================

class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    """
    Supervised Contrastive Loss.
    
    Pulls together embeddings of the same class while pushing apart
    embeddings of different classes.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        name: str = 'supervised_contrastive_loss'
    ):
        super().__init__(name=name)
        self.temperature = temperature
        
    def call(self, labels, embeddings):
        """
        Compute supervised contrastive loss.
        
        Args:
            labels: Shape (batch_size,) - class labels
            embeddings: Shape (batch_size, embedding_dim)
        """
        # Normalize embeddings
        embeddings = tf.math.l2_normalize(embeddings, axis=1)
        
        # Compute similarity matrix
        similarity = tf.matmul(embeddings, embeddings, transpose_b=True)
        similarity = similarity / self.temperature
        
        # Create masks
        batch_size = tf.shape(labels)[0]
        labels_expand = tf.expand_dims(labels, 1)
        mask_same = tf.cast(
            tf.equal(labels_expand, tf.transpose(labels_expand)),
            tf.float32
        )
        
        # Remove self-similarity
        mask_self = tf.eye(batch_size, dtype=tf.float32)
        mask_positives = mask_same - mask_self
        
        # Compute log softmax
        exp_similarity = tf.exp(similarity - tf.reduce_max(similarity, axis=1, keepdims=True))
        
        # Mask out self
        exp_similarity = exp_similarity * (1 - mask_self)
        
        # Sum over all (including negatives)
        sum_exp = tf.reduce_sum(exp_similarity, axis=1, keepdims=True)
        
        # Log probability of positives
        log_prob = similarity - tf.math.log(sum_exp + 1e-8)
        
        # Mask and average over positives
        num_positives = tf.reduce_sum(mask_positives, axis=1)
        num_positives = tf.maximum(num_positives, 1.0)  # Avoid division by zero
        
        loss_per_sample = -tf.reduce_sum(mask_positives * log_prob, axis=1) / num_positives
        
        return tf.reduce_mean(loss_per_sample)


class TimeDiffLoss(tf.keras.losses.Loss):
    """
    Time-aware loss using frc_diff information.
    
    Encourages embeddings of samples close to RLF events to be more clustered,
    while samples far from RLF can be more spread out.
    
    Key insight: Samples 1.2 sec apart can have similar inputs but different labels.
    This loss uses time_diff to modulate the embedding space.
    """
    
    def __init__(
        self,
        frc_per_second: int = FRC_PER_SECOND,
        margin: float = 0.5,
        name: str = 'time_diff_loss'
    ):
        super().__init__(name=name)
        self.frc_per_second = float(frc_per_second)
        self.margin = margin
        
    def call(self, y_true, y_pred):
        """
        Compute time-diff aware loss.
        
        Args:
            y_true: Tuple of (labels, time_diffs, valid_mask)
                - labels: Shape (batch_size,) - class labels
                - time_diffs: Shape (batch_size,) - time diff in frc units
                - valid_mask: Shape (batch_size,) - 1 if time_diff is valid, 0 otherwise
            y_pred: Shape (batch_size, embedding_dim) - embeddings
        """
        labels, time_diffs, valid_mask = y_true
        embeddings = y_pred
        
        # Convert time_diffs to seconds
        time_seconds = time_diffs / self.frc_per_second
        
        # Normalize embeddings
        embeddings_norm = tf.math.l2_normalize(embeddings, axis=1)
        
        # Compute pairwise distances
        distances = tf.reduce_sum(
            tf.square(
                tf.expand_dims(embeddings_norm, 1) - 
                tf.expand_dims(embeddings_norm, 0)
            ),
            axis=2
        )
        
        # For samples close to RLF (small time_diff), enforce smaller distances
        # to samples with same label
        batch_size = tf.shape(labels)[0]
        
        # Create label similarity mask
        labels_expand = tf.expand_dims(labels, 1)
        same_label = tf.cast(
            tf.equal(labels_expand, tf.transpose(labels_expand)),
            tf.float32
        )
        diff_label = 1.0 - same_label
        
        # Create time-based weights
        # Samples closer to RLF should have stronger clustering
        time_weight = tf.exp(-time_seconds / 0.5)  # Decay with 0.5 sec time constant
        time_weight = time_weight * tf.cast(valid_mask, tf.float32)
        time_weight = tf.expand_dims(time_weight, 1) * tf.expand_dims(time_weight, 0)
        
        # Valid pair mask
        valid_pair = tf.expand_dims(valid_mask, 1) * tf.expand_dims(valid_mask, 0)
        valid_pair = tf.cast(valid_pair, tf.float32)
        
        # Mask out diagonal
        mask_diag = 1.0 - tf.eye(batch_size, dtype=tf.float32)
        
        # Loss for same-label pairs: minimize distance (weighted by time proximity)
        same_label_loss = distances * same_label * time_weight * mask_diag
        
        # Loss for different-label pairs: maximize distance with margin
        margin_loss = tf.maximum(0.0, self.margin - distances)
        diff_label_loss = margin_loss * diff_label * time_weight * mask_diag
        
        # Combine losses
        total_pairs = tf.reduce_sum(valid_pair * mask_diag) + 1e-8
        loss = (
            tf.reduce_sum(same_label_loss * valid_pair) +
            tf.reduce_sum(diff_label_loss * valid_pair)
        ) / total_pairs
        
        return loss


# =============================================================================
# Model Definition
# =============================================================================

class RLFEmbeddingModel(tf.keras.Model):
    """
    MLP-based Embedding Model for RLF Prediction.
    
    Architecture:
    - Input layer (flattened time series)
    - Hidden layers with optional batch normalization and dropout
    - Embedding layer (intermediate representation)
    - Classification head (4-class output)
    
    Losses:
    - L_cls: Weighted cross-entropy for 4-class classification
    - L_contrast: Supervised contrastive loss on embeddings
    - L_time: Time-aware loss using frc_diff
    """
    
    def __init__(self, config: EmbeddingModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Build embedding block
        self.embedding_block = EmbeddingBlock(
            hidden_units=config.hidden_layers,
            embedding_dim=config.embedding_dim,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm,
            l2_reg=config.l2_regularization
        )
        
        # Classification head
        self.classifier = tf.keras.layers.Dense(
            config.num_classes,
            activation='softmax',
            name='classifier'
        )
        
        # Loss functions
        self.contrastive_loss_fn = SupervisedContrastiveLoss(
            temperature=config.contrast_temperature
        )
        self.time_loss_fn = TimeDiffLoss(frc_per_second=config.frc_per_second)
        
        # Loss weights
        self.cls_loss_weight = config.cls_loss_weight
        self.contrast_loss_weight = config.contrast_loss_weight
        self.time_loss_weight = config.time_loss_weight
        
        # Class weights for weighted cross-entropy
        self.class_weights = tf.constant(config.class_weights, dtype=tf.float32)
        
    def call(self, inputs, training=False):
        """
        Forward pass.
        
        Args:
            inputs: Shape (batch_size, input_dim)
            training: Whether in training mode
            
        Returns:
            Tuple of (class_probabilities, embeddings)
        """
        embeddings = self.embedding_block(inputs, training=training)
        class_probs = self.classifier(embeddings)
        return class_probs, embeddings
    
    def get_embeddings(self, inputs, training=False):
        """Get embeddings only."""
        return self.embedding_block(inputs, training=training)
    
    def compute_loss(
        self,
        x: tf.Tensor,
        y_labels: tf.Tensor,
        time_diffs: Optional[tf.Tensor] = None,
        valid_time_mask: Optional[tf.Tensor] = None,
        training: bool = True
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            x: Input features (batch_size, input_dim)
            y_labels: Class labels (batch_size,) - values 0, 1, 2, 3
            time_diffs: Time diff to nearest RLF in frc units (batch_size,)
            valid_time_mask: Boolean mask for valid time_diffs (batch_size,)
            training: Whether in training mode
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Forward pass
        class_probs, embeddings = self(x, training=training)
        
        # Classification loss with class weights
        weights = tf.gather(self.class_weights, y_labels)
        cls_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_labels, class_probs
        )
        cls_loss = tf.reduce_mean(cls_loss * weights)
        
        # Contrastive loss
        contrast_loss = self.contrastive_loss_fn(y_labels, embeddings)
        
        # Time-aware loss
        if time_diffs is not None and valid_time_mask is not None:
            time_loss = self.time_loss_fn(
                (y_labels, time_diffs, valid_time_mask),
                embeddings
            )
        else:
            time_loss = tf.constant(0.0)
        
        # Combined loss
        total_loss = (
            self.cls_loss_weight * cls_loss +
            self.contrast_loss_weight * contrast_loss +
            self.time_loss_weight * time_loss
        )
        
        loss_dict = {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'contrast_loss': contrast_loss,
            'time_loss': time_loss
        }
        
        return total_loss, loss_dict
    
    def train_step_custom(
        self,
        x: tf.Tensor,
        y_labels: tf.Tensor,
        time_diffs: Optional[tf.Tensor] = None,
        valid_time_mask: Optional[tf.Tensor] = None,
        optimizer: tf.keras.optimizers.Optimizer = None
    ) -> Dict[str, tf.Tensor]:
        """
        Custom training step.
        
        Returns:
            Dictionary of loss values
        """
        with tf.GradientTape() as tape:
            total_loss, loss_dict = self.compute_loss(
                x, y_labels, time_diffs, valid_time_mask, training=True
            )
        
        gradients = tape.gradient(total_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss_dict


def create_embedding_model(
    input_dim: int = DEFAULT_INPUT_DIM,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    hidden_layers: List[int] = None,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    contrast_temperature: float = DEFAULT_CONTRAST_TEMPERATURE,
    time_loss_weight: float = DEFAULT_TIME_LOSS_WEIGHT,
    contrast_loss_weight: float = DEFAULT_CONTRAST_LOSS_WEIGHT,
    cls_loss_weight: float = DEFAULT_CLS_LOSS_WEIGHT,
    class_weights: List[float] = None,
    use_batch_norm: bool = True,
    l2_regularization: float = 0.0001
) -> Tuple[RLFEmbeddingModel, tf.keras.optimizers.Optimizer]:
    """
    Factory function to create an embedding model with optimizer.
    
    Returns:
        Tuple of (model, optimizer)
    """
    if hidden_layers is None:
        hidden_layers = [128, 64]
    
    config = EmbeddingModelConfig(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        contrast_temperature=contrast_temperature,
        time_loss_weight=time_loss_weight,
        contrast_loss_weight=contrast_loss_weight,
        cls_loss_weight=cls_loss_weight,
        class_weights=class_weights,
        use_batch_norm=use_batch_norm,
        l2_regularization=l2_regularization
    )
    
    model = RLFEmbeddingModel(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Build model by calling with dummy input
    dummy_input = tf.zeros((1, input_dim))
    _ = model(dummy_input)
    
    return model, optimizer


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        y_proba: Prediction probabilities (N, num_classes)
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        confusion_matrix, precision_score, recall_score, 
        f1_score, accuracy_score
    )
    
    # 4-class metrics
    cm_4class = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    accuracy = accuracy_score(y_true, y_pred)
    
    # 2-class metrics (RLF vs Normal)
    # RLF: class 0, 1, 2 -> binary 1
    # Normal: class 3 -> binary 0
    y_true_binary = (y_true != 3).astype(int)
    y_pred_binary = (y_pred != 3).astype(int)
    
    cm_2class = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    
    # TN, FP, FN, TP for 2-class
    tn = cm_2class[0, 0]
    fp = cm_2class[0, 1]
    fn = cm_2class[1, 0]
    tp = cm_2class[1, 1]
    
    # Recall (sensitivity) for RLF detection
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # False Alarm Rate = FP / (FP + TN) = 1 - specificity
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        'accuracy_4class': accuracy,
        'confusion_matrix_4class': cm_4class,
        'confusion_matrix_2class': cm_2class,
        'recall_2class': recall,
        'precision_2class': precision,
        'false_alarm_rate': far,
        'f1_2class': f1,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }
    
    return metrics


def print_metrics(metrics: Dict[str, Any], prefix: str = "") -> None:
    """Print metrics in a formatted way."""
    print(f"{prefix}4-Class Accuracy: {metrics['accuracy_4class']:.4f}")
    print(f"{prefix}2-Class Recall (RLF): {metrics['recall_2class']:.4f}")
    print(f"{prefix}2-Class Precision: {metrics['precision_2class']:.4f}")
    print(f"{prefix}False Alarm Rate: {metrics['false_alarm_rate']:.4f}")
    print(f"{prefix}F1 Score: {metrics['f1_2class']:.4f}")
    print(f"{prefix}2-Class Confusion Matrix:")
    print(f"{prefix}  [[TN={metrics['tn']}, FP={metrics['fp']}]")
    print(f"{prefix}   [FN={metrics['fn']}, TP={metrics['tp']}]]")


if __name__ == "__main__":
    # Demo
    print("Testing RLF Embedding Model")
    print("=" * 60)
    
    # Create model
    model, optimizer = create_embedding_model(
        input_dim=350,
        embedding_dim=16,
        hidden_layers=[128, 64, 32],
        dropout_rate=0.3
    )
    
    # Print model summary
    model.summary()
    
    # Test with dummy data
    batch_size = 32
    x = np.random.randn(batch_size, 350).astype(np.float32)
    y = np.random.randint(0, 4, size=(batch_size,))
    time_diffs = np.random.uniform(0, 15625 * 2, size=(batch_size,)).astype(np.float32)
    valid_mask = np.random.randint(0, 2, size=(batch_size,)).astype(np.float32)
    
    # Forward pass
    class_probs, embeddings = model(x, training=False)
    print(f"\nClass probabilities shape: {class_probs.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Compute loss
    total_loss, loss_dict = model.compute_loss(
        tf.constant(x),
        tf.constant(y),
        tf.constant(time_diffs),
        tf.constant(valid_mask),
        training=True
    )
    
    print(f"\nLoss values:")
    for name, value in loss_dict.items():
        print(f"  {name}: {value.numpy():.4f}")
