import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class Visualizer:
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if not HAS_MATPLOTLIB:
            logging.warning("Matplotlib not available. Visualization disabled.")
    
    def plot_training_loss(
        self,
        train_losses: List[float],
        title: str = "Training Loss Curve",
        save_path: Optional[str] = None
    ):
        if not HAS_MATPLOTLIB:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label='Train Loss', linewidth=2, color='steelblue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Training loss plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_score_distribution(
        self,
        scores: np.ndarray,
        title: str = "Anomaly Score Distribution",
        save_path: Optional[str] = None
    ):
        if not HAS_MATPLOTLIB:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax = axes[0]
        ax.hist(scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Score Distribution (Histogram)')
        ax.axvline(np.percentile(scores, 95), color='red', linestyle='--', 
                   label=f'95th percentile: {np.percentile(scores, 95):.4f}')
        ax.axvline(np.percentile(scores, 99), color='orange', linestyle='--',
                   label=f'99th percentile: {np.percentile(scores, 99):.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        bp = ax.boxplot(scores, vert=True)
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Score Distribution (Boxplot)')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Score distribution plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_node_degree_vs_score(
        self,
        node_degrees: np.ndarray,
        node_scores: np.ndarray,
        title: str = "Node Degree vs Anomaly Score",
        save_path: Optional[str] = None
    ):
        if not HAS_MATPLOTLIB:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(node_degrees, node_scores, alpha=0.3, s=10, c='coral')
        ax.set_xlabel('Node Degree')
        ax.set_ylabel('Anomaly Score (Reconstruction Error)')
        ax.set_title(title)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Degree vs score plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_top_anomalies(
        self,
        top_indices: np.ndarray,
        top_scores: np.ndarray,
        title: str = "Top Anomalies",
        save_path: Optional[str] = None
    ):
        if not HAS_MATPLOTLIB:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        k = len(top_indices)
        x = range(k)
        
        bars = ax.bar(x, top_scores, color='salmon', edgecolor='darkred', alpha=0.8)
        ax.set_xlabel('Rank')
        ax.set_ylabel('Anomaly Score')
        ax.set_title(f'{title} (Top {k})')
        ax.grid(True, alpha=0.3, axis='y')
        
        if k > 20:
            step = k // 10
            ax.set_xticks(x[::step])
            ax.set_xticklabels([str(i+1) for i in x[::step]])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Top anomalies plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_embeddings_tsne(
        self,
        embeddings: np.ndarray,
        scores: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        sample_size: int = 5000,
        title: str = "Node Embeddings (t-SNE)",
        save_path: Optional[str] = None
    ):
        if not HAS_MATPLOTLIB:
            return
        
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            logging.warning("sklearn not available. t-SNE visualization skipped.")
            return
        
        n_samples = min(sample_size, len(embeddings))
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        emb_sample = embeddings[indices]
        
        logging.info(f"Performing t-SNE on {n_samples} nodes...")
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb_2d = tsne.fit_transform(emb_sample)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if scores is not None:
            scores_sample = scores[indices]
            scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                               c=scores_sample, cmap='coolwarm', 
                               alpha=0.6, s=15)
            plt.colorbar(scatter, ax=ax, label='Anomaly Score')
        elif labels is not None:
            labels_sample = labels[indices]
            scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
                               c=labels_sample, cmap='tab10',
                               alpha=0.6, s=15)
            plt.colorbar(scatter, ax=ax, label='Label')
        else:
            ax.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.6, s=15, color='steelblue')
        
        ax.set_title(title)
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"t-SNE plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_comprehensive_report(
        self,
        train_losses: List[float],
        node_scores: np.ndarray,
        edge_scores: np.ndarray,
        node_degrees: np.ndarray,
        title: str = "GraphMAE Fraud Detection Report",
        save_path: Optional[str] = None
    ):
        if not HAS_MATPLOTLIB:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        ax = axes[0, 0]
        ax.plot(train_losses, label='Train Loss', linewidth=2, color='steelblue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.hist(node_scores, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Node Anomaly Score Distribution')
        ax.axvline(np.percentile(node_scores, 95), color='red', linestyle='--', 
                   label='95th percentile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        ax.hist(edge_scores, bins=50, alpha=0.7, color='coral', edgecolor='black')
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Edge Anomaly Score Distribution')
        ax.axvline(np.percentile(edge_scores, 95), color='red', linestyle='--',
                   label='95th percentile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        ax.scatter(node_degrees, node_scores, alpha=0.3, s=10, c='coral')
        ax.set_xlabel('Node Degree')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Node Degree vs Anomaly Score')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        percentiles = [50, 75, 90, 95, 99]
        node_pcts = [np.percentile(node_scores, p) for p in percentiles]
        edge_pcts = [np.percentile(edge_scores, p) for p in percentiles]
        
        x = np.arange(len(percentiles))
        width = 0.35
        ax.bar(x - width/2, node_pcts, width, label='Node Scores', color='purple', alpha=0.7)
        ax.bar(x + width/2, edge_pcts, width, label='Edge Scores', color='coral', alpha=0.7)
        ax.set_xlabel('Percentile')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Score Percentiles Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{p}th' for p in percentiles])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""
        Summary Statistics
        ==================
        
        Training:
          - Final Loss: {train_losses[-1]:.4f}
          - Best Loss: {min(train_losses):.4f}
          - Epochs: {len(train_losses)}
        
        Node Anomaly Scores:
          - Mean: {np.mean(node_scores):.4f}
          - Std: {np.std(node_scores):.4f}
          - Max: {np.max(node_scores):.4f}
        
        Edge Anomaly Scores:
          - Mean: {np.mean(edge_scores):.4f}
          - Std: {np.std(edge_scores):.4f}
          - Max: {np.max(edge_scores):.4f}
        
        Detected Anomalies (95th percentile):
          - Nodes: {(node_scores > np.percentile(node_scores, 95)).sum():,}
          - Edges: {(edge_scores > np.percentile(edge_scores, 95)).sum():,}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Statistics Summary')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Comprehensive report saved to {save_path}")
        
        plt.show()
        plt.close()


def create_visualizer(output_dir: str = "./output") -> Visualizer:
    return Visualizer(output_dir)
