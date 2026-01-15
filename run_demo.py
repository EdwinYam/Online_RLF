"""
Demo Script for RLF Prediction System

This script demonstrates:
1. Online Streaming KNN with delayed label updates
2. Embedding model training with Optuna optimization
3. End-to-end inference pipeline

Usage:
    python run_demo.py --mode knn        # Demo streaming KNN
    python run_demo.py --mode optuna     # Run Optuna search (synthetic data)
    python run_demo.py --mode full       # Run full demo
"""

import os
import sys
import argparse
import numpy as np
from typing import Optional

# Try to load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def demo_streaming_knn(
    n_samples: int = 200,
    embedding_dim: int = 16,
    rlf_ratio: float = 0.1
):
    """Demonstrate the Online Streaming KNN functionality."""
    from streaming_knn import (
        OnlineStreamingKNN, KNNConfig, DistanceMetric, VotingScheme,
        StreamingKNNSimulator, FRC_PER_SECOND
    )
    
    print("=" * 60)
    print("Online Streaming KNN Demo")
    print("=" * 60)
    
    # Configuration from environment or defaults
    max_samples = int(os.getenv('KNN_MAX_SAMPLES', 500))
    delay_seconds = float(os.getenv('DELAY_SECONDS', 0.9))
    k_neighbors = int(os.getenv('KNN_K_NEIGHBORS', 5))
    rlf_class_weight = float(os.getenv('KNN_RLF_CLASS_WEIGHT', 3.0))
    
    print(f"\nConfiguration:")
    print(f"  Max samples: {max_samples}")
    print(f"  Delay window: {delay_seconds}s")
    print(f"  K neighbors: {k_neighbors}")
    print(f"  RLF class weight: {rlf_class_weight}")
    
    # Create KNN
    config = KNNConfig(
        max_samples=max_samples,
        delay_seconds=delay_seconds,
        k_neighbors=k_neighbors,
        embedding_dim=embedding_dim,
        distance_metric=DistanceMetric.EUCLIDEAN,
        voting_scheme=VotingScheme.DISTANCE,
        rlf_class_weight=rlf_class_weight,
        normal_class_weight=1.0,
        min_rlf_ratio=0.1,
        max_rlf_ratio=0.5
    )
    
    knn = OnlineStreamingKNN(config)
    simulator = StreamingKNNSimulator(knn)
    
    # Generate synthetic data
    np.random.seed(42)
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    
    # Generate labels with imbalance
    labels = np.zeros(n_samples, dtype=int)
    num_rlf = int(n_samples * rlf_ratio)
    rlf_indices = np.random.choice(n_samples, num_rlf, replace=False)
    labels[rlf_indices] = 1
    
    # Make RLF samples slightly different in embedding space
    embeddings[rlf_indices] += 0.5
    
    print(f"\nData:")
    print(f"  Total samples: {n_samples}")
    print(f"  RLF samples: {num_rlf} ({rlf_ratio*100:.1f}%)")
    print(f"  Embedding dim: {embedding_dim}")
    
    # Run simulation
    print("\nRunning simulation...")
    results = simulator.run_batch(embeddings, labels, time_step_seconds=0.3)
    
    print(f"\nResults:")
    print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  RLF Recall: {results['rlf_recall']:.3f}")
    print(f"  False Alarm Rate: {results['false_alarm_rate']:.3f}")
    
    print(f"\nKNN Statistics:")
    stats = results['knn_stats']
    print(f"  Total samples in storage: {stats['total_samples']}")
    print(f"  RLF samples: {stats['rlf_samples']}")
    print(f"  Normal samples: {stats['normal_samples']}")
    print(f"  RLF ratio: {stats['rlf_ratio']:.3f}")
    print(f"  Pending samples: {stats['pending_samples']}")
    
    return results


def demo_optuna_search(
    n_trials: int = 5,
    n_samples: int = 1000,
    output_dir: str = "results/demo_optuna"
):
    """Demonstrate Optuna hyperparameter search with synthetic data."""
    from embedding_optuna import OptunaConfig, run_optuna_search
    
    print("=" * 60)
    print("Optuna Hyperparameter Search Demo")
    print("=" * 60)
    
    # Configuration
    min_far = float(os.getenv('MIN_FAR', 0.02))
    batch_size = int(os.getenv('BATCH_SIZE', 64))
    max_epochs = int(os.getenv('MAX_EPOCHS', 50))
    patience = int(os.getenv('PATIENCE', 5))
    seed = int(os.getenv('RANDOM_SEED', 42))
    
    print(f"\nConfiguration:")
    print(f"  N trials: {n_trials}")
    print(f"  Min FAR: {min_far}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Patience: {patience}")
    print(f"  Output dir: {output_dir}")
    
    config = OptunaConfig(
        n_trials=n_trials,
        min_far=min_far,
        output_dir=output_dir,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        random_seed=seed
    )
    
    # Run search with synthetic data
    study, summary = run_optuna_search(
        csv_paths=None,
        config=config,
        use_synthetic=True
    )
    
    print(f"\nBest Trial: {summary['best_trial']}")
    print(f"Best Recall: {summary['best_value']:.4f}")
    print(f"\nResults saved to: {output_dir}")
    
    return study, summary


def demo_inference_pipeline(
    model_dir: Optional[str] = None,
    embedding_dim: int = 16
):
    """Demonstrate end-to-end inference pipeline."""
    import tensorflow as tf
    from embedding_model import create_embedding_model
    from streaming_knn import OnlineStreamingKNN, KNNConfig, DistanceMetric, VotingScheme
    
    print("=" * 60)
    print("End-to-End Inference Pipeline Demo")
    print("=" * 60)
    
    # Create embedding model (or load from checkpoint)
    input_dim = 350  # 35 time steps x 10 features
    
    if model_dir and os.path.exists(os.path.join(model_dir, "model_weights.index")):
        print(f"Loading model from {model_dir}")
        model, _ = create_embedding_model(
            input_dim=input_dim,
            embedding_dim=embedding_dim
        )
        model.load_weights(os.path.join(model_dir, "model_weights"))
    else:
        print("Creating new model (no checkpoint found)")
        model, _ = create_embedding_model(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_layers=[128, 64, 32],
            dropout_rate=0.3
        )
    
    # Create KNN
    config = KNNConfig(
        max_samples=500,
        delay_seconds=0.9,
        k_neighbors=5,
        embedding_dim=embedding_dim,
        distance_metric=DistanceMetric.EUCLIDEAN,
        voting_scheme=VotingScheme.DISTANCE,
        rlf_class_weight=3.0
    )
    knn = OnlineStreamingKNN(config)
    
    # Simulate streaming data
    print("\nSimulating streaming inference...")
    np.random.seed(42)
    n_samples = 100
    
    # Simulate data stream
    predictions = []
    confidences = []
    current_time = 0
    time_step = 0.3 * config.frc_per_second  # 0.3 sec per sample
    
    # Pre-fill KNN with some initial samples
    print("Pre-filling KNN with initial samples...")
    init_samples = 50
    for i in range(init_samples):
        # Generate random input
        x = np.random.randn(1, input_dim).astype(np.float32)
        
        # Get embedding from model
        _, embedding = model(tf.constant(x), training=False)
        embedding = embedding.numpy()[0]
        
        # Add to KNN (assuming we know labels for pre-fill)
        label = 1 if np.random.rand() < 0.1 else 0
        from streaming_knn import Sample
        sample = Sample(
            embedding=embedding,
            label=label,
            timestamp=current_time,
            weight=1.0
        )
        knn._add_to_storage(sample)
        current_time += time_step
    
    print(f"KNN initialized with {knn.total_samples} samples")
    
    # Now run inference
    print("\nRunning inference on new samples...")
    for i in range(n_samples):
        # Generate random input
        x = np.random.randn(1, input_dim).astype(np.float32)
        
        # Get embedding from model
        class_probs, embedding = model(tf.constant(x), training=False)
        embedding = embedding.numpy()[0]
        
        # Predict using KNN
        pred, conf, details = knn.predict(embedding, current_time)
        predictions.append(pred)
        confidences.append(conf)
        
        # Push to pending queue
        knn.push(embedding, current_time)
        
        current_time += time_step
    
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    
    print(f"\nInference Results:")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  RLF predictions: {np.sum(predictions == 1)}")
    print(f"  Normal predictions: {np.sum(predictions == 0)}")
    print(f"  Avg confidence: {np.mean(confidences):.3f}")
    
    print(f"\nFinal KNN State:")
    stats = knn.get_statistics()
    print(f"  Samples in storage: {stats['total_samples']}")
    print(f"  Pending samples: {stats['pending_samples']}")
    print(f"  Total predictions made: {stats['total_predictions']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='RLF Prediction System Demo'
    )
    
    parser.add_argument(
        '--mode',
        choices=['knn', 'optuna', 'inference', 'full'],
        default='full',
        help='Demo mode: knn, optuna, inference, or full (all demos)'
    )
    
    parser.add_argument(
        '--n_trials',
        type=int,
        default=5,
        help='Number of Optuna trials for demo (default: 5)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/demo_optuna',
        help='Output directory for Optuna results'
    )
    
    args = parser.parse_args()
    
    if args.mode in ['knn', 'full']:
        demo_streaming_knn()
        print("\n")
    
    if args.mode in ['optuna', 'full']:
        demo_optuna_search(
            n_trials=args.n_trials,
            output_dir=args.output_dir
        )
        print("\n")
    
    if args.mode in ['inference', 'full']:
        demo_inference_pipeline()


if __name__ == "__main__":
    main()
