"""
Online Streaming KNN with Delayed Label Update for RLF Prediction

This module implements a streaming KNN classifier designed for real-time RLF 
(Radio Link Failure) prediction. It handles:
- Delayed label updates (labels only available after a configurable delay window)
- Class imbalance (RLF events are rare)
- Dynamic sample pool management with configurable size limits
- Multiple distance metrics and weighting schemes

Key constraint: Only after delay_window (default 0.9 sec) do we know the true label,
so samples are queued and added to KNN storage only after the delay period.

Time unit: 15625 frc = 1 second
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import heapq


# =============================================================================
# Configuration Constants (all configurable, no hardcoded values)
# =============================================================================

# Time conversion: 15625 frc units = 1 second
FRC_PER_SECOND = 15625

# Default configuration values
DEFAULT_MAX_SAMPLES = 500
DEFAULT_DELAY_SECONDS = 0.9
DEFAULT_K_NEIGHBORS = 5
DEFAULT_EMBEDDING_DIM = 16
DEFAULT_RLF_CLASS_WEIGHT = 1.0
DEFAULT_NORMAL_CLASS_WEIGHT = 1.0


class DistanceMetric(Enum):
    """Supported distance metrics for KNN."""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MANHATTAN = "manhattan"


class VotingScheme(Enum):
    """Voting schemes for KNN prediction."""
    UNIFORM = "uniform"  # Equal weight for all neighbors
    DISTANCE = "distance"  # Inverse distance weighting
    RANKED = "ranked"  # Weight by rank (1/rank)


@dataclass
class KNNConfig:
    """Configuration for Online Streaming KNN."""
    max_samples: int = DEFAULT_MAX_SAMPLES
    delay_seconds: float = DEFAULT_DELAY_SECONDS
    k_neighbors: int = DEFAULT_K_NEIGHBORS
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    voting_scheme: VotingScheme = VotingScheme.DISTANCE
    rlf_class_weight: float = DEFAULT_RLF_CLASS_WEIGHT
    normal_class_weight: float = DEFAULT_NORMAL_CLASS_WEIGHT
    min_rlf_ratio: float = 0.1  # Minimum ratio of RLF samples to maintain
    max_rlf_ratio: float = 0.5  # Maximum ratio of RLF samples
    time_decay_factor: float = 0.99  # Decay factor for older samples (per second)
    frc_per_second: int = FRC_PER_SECOND

    @property
    def delay_frc(self) -> float:
        """Delay window in FRC units."""
        return self.delay_seconds * self.frc_per_second


@dataclass
class Sample:
    """A single sample in the KNN storage."""
    embedding: np.ndarray
    label: int  # 0 = Normal (no RLF), 1 = RLF
    timestamp: float  # FRC timestamp
    weight: float = 1.0
    
    def __lt__(self, other):
        """For heap operations - older samples have lower priority."""
        return self.timestamp < other.timestamp


@dataclass
class PendingSample:
    """A sample waiting for its label (in the delay queue)."""
    embedding: np.ndarray
    timestamp: float
    label: Optional[int] = None  # Will be set after delay
    
    def __lt__(self, other):
        """For heap operations - sorted by timestamp."""
        return self.timestamp < other.timestamp


class OnlineStreamingKNN:
    """
    Online KNN classifier with streaming data and delayed label updates.
    
    This class handles:
    1. Real-time prediction using current sample pool
    2. Delayed label assignment (labels available only after delay_window)
    3. Dynamic sample pool management with class balancing
    4. Configurable distance metrics and voting schemes
    
    Usage:
        config = KNNConfig(max_samples=500, delay_seconds=0.9, k_neighbors=5)
        knn = OnlineStreamingKNN(config)
        
        # For each incoming sample:
        prediction, score = knn.predict(embedding, current_timestamp)
        knn.push(embedding, current_timestamp)
        
        # After delay window, when label is known:
        knn.update_label(original_timestamp, true_label)
    """
    
    def __init__(self, config: Optional[KNNConfig] = None):
        """
        Initialize the Online Streaming KNN.
        
        Args:
            config: KNN configuration. If None, uses defaults.
        """
        self.config = config or KNNConfig()
        
        # Sample storage - separate pools for each class for balanced sampling
        self.rlf_samples: List[Sample] = []  # RLF samples (label=1)
        self.normal_samples: List[Sample] = []  # Normal samples (label=0)
        
        # Pending samples queue (waiting for labels)
        self.pending_queue: List[PendingSample] = []
        
        # Statistics
        self.total_predictions = 0
        self.total_updates = 0
        
    @property
    def total_samples(self) -> int:
        """Total number of samples in storage."""
        return len(self.rlf_samples) + len(self.normal_samples)
    
    @property
    def rlf_ratio(self) -> float:
        """Current ratio of RLF samples."""
        if self.total_samples == 0:
            return 0.0
        return len(self.rlf_samples) / self.total_samples
    
    def _compute_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute distance between two embeddings."""
        if self.config.distance_metric == DistanceMetric.EUCLIDEAN:
            return np.linalg.norm(emb1 - emb2)
        elif self.config.distance_metric == DistanceMetric.COSINE:
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 1.0  # Maximum distance for zero vectors
            return 1.0 - np.dot(emb1, emb2) / (norm1 * norm2)
        elif self.config.distance_metric == DistanceMetric.MANHATTAN:
            return np.sum(np.abs(emb1 - emb2))
        else:
            raise ValueError(f"Unknown distance metric: {self.config.distance_metric}")
    
    def _get_time_weight(self, sample_timestamp: float, current_timestamp: float) -> float:
        """Compute time decay weight for a sample."""
        time_diff_seconds = (current_timestamp - sample_timestamp) / self.config.frc_per_second
        if time_diff_seconds < 0:
            time_diff_seconds = 0
        return self.config.time_decay_factor ** time_diff_seconds
    
    def _get_all_samples(self) -> List[Sample]:
        """Get all samples from both pools."""
        return self.rlf_samples + self.normal_samples
    
    def predict(
        self, 
        embedding: np.ndarray, 
        current_timestamp: float
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        Predict whether RLF will occur within the prediction window.
        
        Args:
            embedding: The embedding vector (dim < 16)
            current_timestamp: Current timestamp in FRC units
            
        Returns:
            Tuple of (prediction, confidence_score, details)
            - prediction: 0 = No RLF, 1 = RLF
            - confidence_score: Confidence of prediction [0, 1]
            - details: Dictionary with additional information
        """
        self.total_predictions += 1
        
        all_samples = self._get_all_samples()
        
        # If not enough samples, return default prediction
        if len(all_samples) < self.config.k_neighbors:
            return 0, 0.5, {
                "status": "insufficient_samples",
                "num_samples": len(all_samples),
                "k_required": self.config.k_neighbors
            }
        
        # Compute distances to all samples
        distances_and_samples = []
        for sample in all_samples:
            dist = self._compute_distance(embedding, sample.embedding)
            time_weight = self._get_time_weight(sample.timestamp, current_timestamp)
            distances_and_samples.append((dist, sample, time_weight))
        
        # Sort by distance and get k nearest neighbors
        distances_and_samples.sort(key=lambda x: x[0])
        k_nearest = distances_and_samples[:self.config.k_neighbors]
        
        # Compute weighted votes
        rlf_vote = 0.0
        normal_vote = 0.0
        
        for rank, (dist, sample, time_weight) in enumerate(k_nearest, 1):
            # Compute vote weight based on voting scheme
            if self.config.voting_scheme == VotingScheme.UNIFORM:
                vote_weight = 1.0
            elif self.config.voting_scheme == VotingScheme.DISTANCE:
                vote_weight = 1.0 / (dist + 1e-8)  # Avoid division by zero
            elif self.config.voting_scheme == VotingScheme.RANKED:
                vote_weight = 1.0 / rank
            else:
                vote_weight = 1.0
            
            # Apply time decay and sample weight
            vote_weight *= time_weight * sample.weight
            
            # Apply class weights
            if sample.label == 1:  # RLF
                vote_weight *= self.config.rlf_class_weight
                rlf_vote += vote_weight
            else:  # Normal
                vote_weight *= self.config.normal_class_weight
                normal_vote += vote_weight
        
        # Compute prediction and confidence
        total_vote = rlf_vote + normal_vote
        if total_vote == 0:
            return 0, 0.5, {"status": "zero_votes"}
        
        rlf_probability = rlf_vote / total_vote
        prediction = 1 if rlf_probability >= 0.5 else 0
        confidence = max(rlf_probability, 1 - rlf_probability)
        
        details = {
            "status": "ok",
            "rlf_probability": rlf_probability,
            "rlf_vote": rlf_vote,
            "normal_vote": normal_vote,
            "k_used": len(k_nearest),
            "avg_distance": np.mean([d[0] for d in k_nearest]),
            "neighbor_labels": [s.label for _, s, _ in k_nearest]
        }
        
        return prediction, confidence, details
    
    def push(self, embedding: np.ndarray, timestamp: float) -> None:
        """
        Push a new sample into the pending queue.
        
        The sample will be added to the KNN storage only after the delay window
        when update_label() is called.
        
        Args:
            embedding: The embedding vector
            timestamp: Timestamp in FRC units
        """
        pending = PendingSample(
            embedding=embedding.copy(),
            timestamp=timestamp,
            label=None
        )
        heapq.heappush(self.pending_queue, pending)
    
    def update_label(self, timestamp: float, label: int) -> bool:
        """
        Update the label for a pending sample after the delay window.
        
        Args:
            timestamp: Original timestamp of the sample
            label: True label (0 = Normal, 1 = RLF)
            
        Returns:
            True if sample was found and updated, False otherwise
        """
        # Find the sample in pending queue
        for i, pending in enumerate(self.pending_queue):
            if abs(pending.timestamp - timestamp) < 1.0:  # Small tolerance
                # Remove from pending queue
                self.pending_queue.pop(i)
                heapq.heapify(self.pending_queue)
                
                # Create sample and add to storage
                sample = Sample(
                    embedding=pending.embedding,
                    label=label,
                    timestamp=pending.timestamp,
                    weight=1.0
                )
                self._add_to_storage(sample)
                self.total_updates += 1
                return True
        
        return False
    
    def process_ready_samples(
        self, 
        current_timestamp: float, 
        label_provider: callable
    ) -> int:
        """
        Process all pending samples that have passed the delay window.
        
        Args:
            current_timestamp: Current timestamp in FRC units
            label_provider: Callable that takes (embedding, timestamp) and returns label
            
        Returns:
            Number of samples processed
        """
        delay_frc = self.config.delay_frc
        processed = 0
        
        while self.pending_queue:
            oldest = self.pending_queue[0]
            if current_timestamp - oldest.timestamp >= delay_frc:
                # Sample is ready - get its label
                heapq.heappop(self.pending_queue)
                label = label_provider(oldest.embedding, oldest.timestamp)
                
                sample = Sample(
                    embedding=oldest.embedding,
                    label=label,
                    timestamp=oldest.timestamp,
                    weight=1.0
                )
                self._add_to_storage(sample)
                processed += 1
                self.total_updates += 1
            else:
                break
        
        return processed
    
    def _add_to_storage(self, sample: Sample) -> None:
        """
        Add a sample to the appropriate storage pool.
        
        Handles class balancing and pool size limits.
        """
        # Add to appropriate pool
        if sample.label == 1:  # RLF
            self.rlf_samples.append(sample)
        else:  # Normal
            self.normal_samples.append(sample)
        
        # Check if we need to evict samples
        self._balance_and_evict()
    
    def _balance_and_evict(self) -> None:
        """Balance class distribution and evict samples if over limit."""
        max_samples = self.config.max_samples
        
        if self.total_samples <= max_samples:
            return
        
        # Calculate target sizes maintaining class balance
        num_to_remove = self.total_samples - max_samples
        
        # Compute current and target ratios
        current_rlf_ratio = self.rlf_ratio
        target_rlf_ratio = max(
            self.config.min_rlf_ratio,
            min(current_rlf_ratio, self.config.max_rlf_ratio)
        )
        
        target_rlf_count = int(max_samples * target_rlf_ratio)
        target_normal_count = max_samples - target_rlf_count
        
        # Evict from pools
        rlf_to_remove = max(0, len(self.rlf_samples) - target_rlf_count)
        normal_to_remove = max(0, len(self.normal_samples) - target_normal_count)
        
        # Remove oldest samples (those at the front after sorting by timestamp)
        if rlf_to_remove > 0:
            self.rlf_samples.sort(key=lambda s: s.timestamp)
            self.rlf_samples = self.rlf_samples[rlf_to_remove:]
        
        if normal_to_remove > 0:
            self.normal_samples.sort(key=lambda s: s.timestamp)
            self.normal_samples = self.normal_samples[normal_to_remove:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics about the KNN state."""
        return {
            "total_samples": self.total_samples,
            "rlf_samples": len(self.rlf_samples),
            "normal_samples": len(self.normal_samples),
            "rlf_ratio": self.rlf_ratio,
            "pending_samples": len(self.pending_queue),
            "total_predictions": self.total_predictions,
            "total_updates": self.total_updates,
            "config": {
                "max_samples": self.config.max_samples,
                "delay_seconds": self.config.delay_seconds,
                "k_neighbors": self.config.k_neighbors,
                "distance_metric": self.config.distance_metric.value,
                "voting_scheme": self.config.voting_scheme.value
            }
        }
    
    def clear(self) -> None:
        """Clear all samples and pending queue."""
        self.rlf_samples.clear()
        self.normal_samples.clear()
        self.pending_queue.clear()
        self.total_predictions = 0
        self.total_updates = 0


class StreamingKNNSimulator:
    """
    Simulator for testing the Online Streaming KNN with sample data.
    
    This class simulates a streaming data source and manages the delay
    queue for realistic testing.
    """
    
    def __init__(
        self, 
        knn: OnlineStreamingKNN,
        frc_per_second: int = FRC_PER_SECOND
    ):
        """
        Initialize the simulator.
        
        Args:
            knn: The OnlineStreamingKNN instance to use
            frc_per_second: FRC units per second
        """
        self.knn = knn
        self.frc_per_second = frc_per_second
        self.current_time = 0.0
        
        # Buffer for samples waiting for labels
        self.label_buffer: Dict[float, int] = {}  # timestamp -> label
        
    def simulate_sample(
        self, 
        embedding: np.ndarray, 
        true_label: int,
        time_step_seconds: float = 0.3
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        Simulate receiving a new sample.
        
        Args:
            embedding: The embedding vector
            true_label: The true label (for later update)
            time_step_seconds: Time step in seconds
            
        Returns:
            Prediction results from KNN
        """
        # Advance time
        self.current_time += time_step_seconds * self.frc_per_second
        
        # Store label for later
        self.label_buffer[self.current_time] = true_label
        
        # Get prediction
        prediction, confidence, details = self.knn.predict(
            embedding, self.current_time
        )
        
        # Push to pending queue
        self.knn.push(embedding, self.current_time)
        
        # Process any samples that are now ready
        delay_frc = self.knn.config.delay_frc
        ready_timestamps = [
            ts for ts in self.label_buffer.keys()
            if self.current_time - ts >= delay_frc
        ]
        
        for ts in ready_timestamps:
            label = self.label_buffer.pop(ts)
            self.knn.update_label(ts, label)
        
        return prediction, confidence, details
    
    def run_batch(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        time_step_seconds: float = 0.3
    ) -> Dict[str, Any]:
        """
        Run simulation on a batch of samples.
        
        Args:
            embeddings: Array of shape (N, embedding_dim)
            labels: Array of shape (N,) with labels
            time_step_seconds: Time step between samples
            
        Returns:
            Dictionary with results and metrics
        """
        predictions = []
        confidences = []
        true_labels = []
        
        for i in range(len(embeddings)):
            pred, conf, _ = self.simulate_sample(
                embeddings[i],
                labels[i],
                time_step_seconds
            )
            predictions.append(pred)
            confidences.append(conf)
            true_labels.append(labels[i])
        
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Compute metrics
        accuracy = np.mean(predictions == true_labels)
        
        # RLF recall (sensitivity)
        rlf_mask = true_labels == 1
        if np.sum(rlf_mask) > 0:
            rlf_recall = np.mean(predictions[rlf_mask] == 1)
        else:
            rlf_recall = 0.0
        
        # False alarm rate (1 - specificity)
        normal_mask = true_labels == 0
        if np.sum(normal_mask) > 0:
            false_alarm_rate = np.mean(predictions[normal_mask] == 1)
        else:
            false_alarm_rate = 0.0
        
        return {
            "predictions": predictions,
            "true_labels": true_labels,
            "confidences": np.array(confidences),
            "accuracy": accuracy,
            "rlf_recall": rlf_recall,
            "false_alarm_rate": false_alarm_rate,
            "knn_stats": self.knn.get_statistics()
        }


# =============================================================================
# Utility Functions
# =============================================================================

def create_default_knn(
    max_samples: int = DEFAULT_MAX_SAMPLES,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    rlf_class_weight: float = DEFAULT_RLF_CLASS_WEIGHT
) -> OnlineStreamingKNN:
    """
    Create an OnlineStreamingKNN with common configuration.
    
    Args:
        max_samples: Maximum number of samples to store
        delay_seconds: Delay window in seconds
        k_neighbors: Number of neighbors for KNN
        rlf_class_weight: Weight for RLF class (to handle imbalance)
        
    Returns:
        Configured OnlineStreamingKNN instance
    """
    config = KNNConfig(
        max_samples=max_samples,
        delay_seconds=delay_seconds,
        k_neighbors=k_neighbors,
        rlf_class_weight=rlf_class_weight,
        distance_metric=DistanceMetric.EUCLIDEAN,
        voting_scheme=VotingScheme.DISTANCE
    )
    return OnlineStreamingKNN(config)


def demo_streaming_knn():
    """Demonstrate the streaming KNN functionality."""
    print("=" * 60)
    print("Online Streaming KNN Demo")
    print("=" * 60)
    
    # Configuration
    embedding_dim = 16
    num_samples = 100
    rlf_ratio = 0.1  # 10% RLF samples
    
    # Create KNN
    knn = create_default_knn(
        max_samples=50,
        delay_seconds=0.9,
        k_neighbors=5,
        rlf_class_weight=3.0  # Higher weight for rare RLF class
    )
    
    # Create simulator
    simulator = StreamingKNNSimulator(knn)
    
    # Generate synthetic data
    np.random.seed(42)
    embeddings = np.random.randn(num_samples, embedding_dim)
    
    # Generate labels with imbalance
    labels = np.zeros(num_samples, dtype=int)
    num_rlf = int(num_samples * rlf_ratio)
    rlf_indices = np.random.choice(num_samples, num_rlf, replace=False)
    labels[rlf_indices] = 1
    
    # Make RLF samples slightly different in embedding space
    embeddings[rlf_indices] += 0.5
    
    # Run simulation
    results = simulator.run_batch(embeddings, labels, time_step_seconds=0.3)
    
    print(f"\nResults:")
    print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  RLF Recall: {results['rlf_recall']:.3f}")
    print(f"  False Alarm Rate: {results['false_alarm_rate']:.3f}")
    print(f"\nKNN Statistics:")
    for key, value in results['knn_stats'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo_streaming_knn()
