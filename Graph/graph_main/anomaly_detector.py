import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from config import AnomalyConfig

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class AnomalyDetector:
    def __init__(self, model, config: AnomalyConfig, device=None):
        self.model = model
        self.config = config
        self.device = device
        
        self.node_scores = None
        self.edge_scores = None
        self.node_embeddings = None
    
    def compute_node_anomaly_scores(
        self, 
        data,
        num_samples: Optional[int] = None
    ) -> np.ndarray:
        num_samples = num_samples or self.config.num_samples
        
        logging.info(f"Computing node anomaly scores (samples: {num_samples})...")
        
        self.node_scores = self.model.compute_node_anomaly_score(
            data, num_samples=num_samples
        ).cpu().numpy()
        
        logging.info(f"Node anomaly scores computed. Shape: {self.node_scores.shape}")
        
        return self.node_scores
    
    def compute_reconstruction_error(self, data) -> np.ndarray:
        logging.info("Computing reconstruction errors...")
        
        self.node_scores = self.model.compute_reconstruction_error(data).cpu().numpy()
        
        logging.info(f"Reconstruction errors computed. Shape: {self.node_scores.shape}")
        
        return self.node_scores
    
    def compute_edge_anomaly_scores(
        self,
        data,
        strategy: Optional[str] = None
    ) -> np.ndarray:
        if self.node_scores is None:
            self.compute_reconstruction_error(data)
        
        strategy = strategy or self.config.edge_score_strategy
        
        if hasattr(data, 'original_edge_index'):
            edge_index = data.original_edge_index.cpu().numpy()
        else:
            edge_index = data.edge_index.cpu().numpy()
        
        src_indices = edge_index[0]
        dst_indices = edge_index[1]
        
        if strategy == 'max':
            self.edge_scores = np.maximum(
                self.node_scores[src_indices],
                self.node_scores[dst_indices]
            )
        elif strategy == 'mean':
            self.edge_scores = (
                self.node_scores[src_indices] + 
                self.node_scores[dst_indices]
            ) / 2
        elif strategy == 'sum':
            self.edge_scores = (
                self.node_scores[src_indices] + 
                self.node_scores[dst_indices]
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logging.info(f"Edge anomaly scores computed (strategy: {strategy}). Shape: {self.edge_scores.shape}")
        
        return self.edge_scores
    
    def get_node_embeddings(self, data) -> np.ndarray:
        logging.info("Getting node embeddings...")
        
        self.node_embeddings = self.model.get_embeddings(data).cpu().numpy()
        
        logging.info(f"Node embeddings obtained. Shape: {self.node_embeddings.shape}")
        
        return self.node_embeddings
    
    def get_top_anomalies(
        self,
        k: int = 100,
        level: str = 'edge'
    ) -> Tuple[np.ndarray, np.ndarray]:
        if level == 'edge':
            if self.edge_scores is None:
                raise ValueError("Edge scores not computed. Call compute_edge_anomaly_scores first.")
            scores = self.edge_scores
        else:
            if self.node_scores is None:
                raise ValueError("Node scores not computed. Call compute_node_anomaly_scores first.")
            scores = self.node_scores
        
        top_indices = np.argsort(scores)[-k:][::-1]
        top_scores = scores[top_indices]
        
        return top_indices, top_scores
    
    def get_anomaly_threshold(
        self,
        percentile: Optional[float] = None,
        level: str = 'edge'
    ) -> float:
        percentile = percentile or self.config.threshold_percentile
        
        if level == 'edge':
            scores = self.edge_scores
        else:
            scores = self.node_scores
        
        if scores is None:
            raise ValueError(f"{level} scores not computed.")
        
        threshold = np.percentile(scores, percentile)
        
        return threshold
    
    def classify_anomalies(
        self,
        threshold: Optional[float] = None,
        level: str = 'edge'
    ) -> np.ndarray:
        if threshold is None:
            threshold = self.get_anomaly_threshold(level=level)
        
        if level == 'edge':
            scores = self.edge_scores
        else:
            scores = self.node_scores
        
        predictions = (scores > threshold).astype(int)
        
        logging.info(f"Classified anomalies (threshold: {threshold:.4f})")
        logging.info(f"  - Total: {len(predictions)}")
        logging.info(f"  - Anomalies: {predictions.sum()}")
        logging.info(f"  - Ratio: {predictions.mean():.4%}")
        
        return predictions


class UnsupervisedEvaluator:
    def __init__(self, detector: AnomalyDetector):
        self.detector = detector
    
    def compute_score_statistics(self, level: str = 'edge') -> Dict:
        if level == 'edge':
            scores = self.detector.edge_scores
        else:
            scores = self.detector.node_scores
        
        if scores is None:
            return {}
        
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'percentiles': {
                '90': float(np.percentile(scores, 90)),
                '95': float(np.percentile(scores, 95)),
                '99': float(np.percentile(scores, 99)),
            }
        }
    
    def analyze_top_k(
        self,
        data,
        k_values: Optional[List[int]] = None,
        level: str = 'edge'
    ) -> Dict:
        k_values = k_values or self.detector.config.top_k_values
        
        results = {}
        for k in k_values:
            indices, scores = self.detector.get_top_anomalies(k, level)
            
            results[f'top_{k}'] = {
                'indices': indices.tolist(),
                'scores': scores.tolist(),
                'mean_score': float(np.mean(scores)),
                'min_score': float(np.min(scores)),
            }
        
        return results
    
    def print_report(self, level: str = 'edge'):
        print("=" * 80)
        print(f"Anomaly Detection Report - {level.upper()} Level")
        print("=" * 80)
        
        stats = self.compute_score_statistics(level)
        
        print(f"\nScore Statistics:")
        print(f"  - Mean: {stats['mean']:.6f}")
        print(f"  - Std: {stats['std']:.6f}")
        print(f"  - Min: {stats['min']:.6f}")
        print(f"  - Max: {stats['max']:.6f}")
        print(f"  - Median: {stats['median']:.6f}")
        
        print(f"\nPercentiles:")
        for p, v in stats['percentiles'].items():
            print(f"  - {p}th: {v:.6f}")
        
        threshold = self.detector.get_anomaly_threshold(level=level)
        print(f"\nAnomaly Threshold ({self.detector.config.threshold_percentile}th percentile): {threshold:.6f}")
        
        predictions = self.detector.classify_anomalies(threshold, level)
        print(f"\nClassification Results:")
        print(f"  - Total: {len(predictions):,}")
        print(f"  - Anomalies: {predictions.sum():,}")
        print(f"  - Anomaly Ratio: {predictions.mean():.4%}")
        
        print("\n" + "=" * 80)


def detect_anomalies(
    model,
    data,
    config: AnomalyConfig,
    level: str = 'edge'
) -> Tuple[AnomalyDetector, np.ndarray]:
    detector = AnomalyDetector(model, config)
    
    detector.compute_reconstruction_error(data)
    
    if level == 'edge':
        scores = detector.compute_edge_anomaly_scores(data)
    else:
        scores = detector.node_scores
    
    evaluator = UnsupervisedEvaluator(detector)
    evaluator.print_report(level)
    
    return detector, scores
