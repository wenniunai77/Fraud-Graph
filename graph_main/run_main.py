import os
import sys
import json
import argparse
import logging
import pickle
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MainConfig
from models import GraphMAE
from trainer import Trainer
from anomaly_detector import AnomalyDetector
from visualization import Visualizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def load_preprocessed_data(config: MainConfig):
    logging.info("Loading preprocessed data...")
    
    if not os.path.exists(config.preprocessed_dir):
        raise FileNotFoundError(
            f"Preprocessed directory not found: {config.preprocessed_dir}\n"
            f"Please run: python preprocess/run_preprocess.py --data_path <your_data.csv>"
        )
    
    graph_path = config.get_preprocessed_path(config.graph_data_file)
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph data file not found: {graph_path}")
    
    data = torch.load(graph_path)
    logging.info(f"  - Graph data loaded: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    
    mapping_path = config.get_preprocessed_path(config.node_mapping_file)
    if os.path.exists(mapping_path):
        with open(mapping_path, 'rb') as f:
            node_mapping = pickle.load(f)
        logging.info(f"  - Node mapping loaded")
    else:
        node_mapping = None
        logging.warning(f"  - Node mapping file not found")
    
    stats_path = config.get_preprocessed_path(config.statistics_file)
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            statistics = json.load(f)
        logging.info(f"  - Statistics loaded")
    else:
        statistics = None
        logging.warning(f"  - Statistics file not found")
    
    return data, node_mapping, statistics


def run_main(config: MainConfig):
    logging.info("=" * 80)
    logging.info("GraphMAE Main Program Start")
    logging.info("=" * 80)
    
    start_time = datetime.now()
    
    set_seed(config.seed)
    
    config.ensure_dirs()
    
    if config.device >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{config.device}')
        logging.info(f"Using GPU: {torch.cuda.get_device_name(config.device)}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    
    logging.info("\n" + "=" * 40)
    logging.info("Step 1: Load Preprocessed Data")
    logging.info("=" * 40)
    
    data, node_mapping, statistics = load_preprocessed_data(config)
    data = data.to(device)
    
    in_channels = data.x.shape[1]
    logging.info(f"Input feature dimension: {in_channels}")
    
    logging.info("\n" + "=" * 40)
    logging.info("Step 2: Build Model")
    logging.info("=" * 40)
    
    model = GraphMAE(
        in_channels=in_channels,
        hidden_channels=config.model.hidden_channels,
        out_channels=config.model.out_channels,
        encoder_type=config.model.encoder_type,
        decoder_type=config.model.decoder_type,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        mask_rate=config.model.mask_rate,
        replace_rate=config.model.replace_rate,
        loss_fn=config.model.loss_fn,
        alpha_l=config.model.alpha_l
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model built:")
    logging.info(f"  - Encoder: {config.model.encoder_type}")
    logging.info(f"  - Decoder: {config.model.decoder_type}")
    logging.info(f"  - Total parameters: {total_params:,}")
    logging.info(f"  - Trainable parameters: {trainable_params:,}")
    
    run_output_dir = config.create_run_output_dir()
    logging.info(f"  - Output directory: {run_output_dir}")
    
    logging.info("\n" + "=" * 40)
    logging.info("Step 3: Train Model")
    logging.info("=" * 40)
    
    trainer = Trainer(model, config.train, device)
    history = trainer.train(data, verbose=config.verbose)
    
    logging.info(f"Training complete:")
    logging.info(f"  - Epochs trained: {history['epochs_trained']}")
    logging.info(f"  - Final loss: {history['train_losses'][-1]:.6f}")
    logging.info(f"  - Best loss: {history['best_loss']:.6f}")
    
    if config.save_model:
        model_path = os.path.join(run_output_dir, "graphmae_model.pt")
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model saved: {model_path}")
    
    logging.info("\n" + "=" * 40)
    logging.info("Step 4: Anomaly Detection")
    logging.info("=" * 40)
    
    detector = AnomalyDetector(model, config.anomaly, device)
    
    node_scores = detector.compute_reconstruction_error(data)
    logging.info(f"Node anomaly scores: shape={node_scores.shape}")
    
    edge_scores = detector.compute_edge_anomaly_scores(data)
    logging.info(f"Edge anomaly scores: shape={edge_scores.shape}")
    
    node_embeddings = detector.get_node_embeddings(data)
    logging.info(f"Node embeddings: shape={node_embeddings.shape}")
    
    threshold = np.percentile(edge_scores, config.anomaly.threshold_percentile)
    num_anomalies = (edge_scores > threshold).sum()
    logging.info(f"Anomaly threshold ({config.anomaly.threshold_percentile}th): {threshold:.6f}")
    logging.info(f"Detected anomalous edges: {num_anomalies} ({num_anomalies/len(edge_scores)*100:.2f}%)")
    
    top_k = 100
    top_indices, top_scores = detector.get_top_anomalies(k=top_k, level='edge')
    logging.info(f"\nTop {top_k} most anomalous transactions:")
    for i in range(min(10, len(top_indices))):
        logging.info(f"  {i+1}. Transaction {top_indices[i]}: score {top_scores[i]:.6f}")
    
    results = {
        "encoder_type": config.model.encoder_type,
        "decoder_type": config.model.decoder_type,
        "node_scores": node_scores.tolist(),
        "edge_scores": edge_scores.tolist(),
        "top_anomaly_indices": top_indices.tolist(),
        "top_anomaly_scores": top_scores.tolist(),
        "threshold": float(threshold),
        "num_anomalies": int(num_anomalies),
        "statistics": {
            "node_score_mean": float(np.mean(node_scores)),
            "node_score_std": float(np.std(node_scores)),
            "edge_score_mean": float(np.mean(edge_scores)),
            "edge_score_std": float(np.std(edge_scores))
        }
    }
    
    results_path = os.path.join(run_output_dir, "anomaly_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Anomaly detection results saved: {results_path}")
    
    if config.visualize:
        logging.info("\n" + "=" * 40)
        logging.info("Step 5: Visualization")
        logging.info("=" * 40)
        
        try:
            from torch_geometric.utils import degree
            node_degrees = degree(
                data.original_edge_index[0], 
                num_nodes=data.num_nodes
            ).cpu().numpy()
        except:
            node_degrees = np.ones(data.num_nodes)
        
        visualizer = Visualizer(run_output_dir)
        
        visualizer.plot_comprehensive_report(
            train_losses=history['train_losses'],
            node_scores=node_scores,
            edge_scores=edge_scores,
            node_degrees=node_degrees,
            save_path=os.path.join(run_output_dir, "comprehensive_report.png")
        )
        
        visualizer.plot_embeddings_tsne(
            embeddings=node_embeddings,
            scores=node_scores,
            sample_size=min(3000, len(node_scores)),
            save_path=os.path.join(run_output_dir, "embeddings_tsne.png")
        )
        
        logging.info("Visualization complete")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logging.info("\n" + "=" * 80)
    logging.info("Main Program Complete")
    logging.info("=" * 80)
    logging.info(f"Total duration: {duration:.2f} seconds")
    logging.info(f"Run output directory: {run_output_dir}")
    
    return {
        "history": history,
        "node_scores": node_scores,
        "edge_scores": edge_scores,
        "node_embeddings": node_embeddings,
        "top_anomalies": (top_indices, top_scores),
        "run_output_dir": run_output_dir
    }


def main():
    parser = argparse.ArgumentParser(description="GraphMAE Main Program")
    
    parser.add_argument(
        "--preprocessed_dir", 
        type=str, 
        default="./processed_data",
        help="Preprocessed data directory"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./output",
        help="Output directory"
    )
    
    parser.add_argument("--encoder_type", type=str, default="gat", help="Encoder type")
    parser.add_argument("--hidden_channels", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--out_channels", type=int, default=128, help="Output dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--mask_rate", type=float, default=0.5, help="Mask rate")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    
    parser.add_argument("--device", type=int, default=0, help="GPU device ID, -1 for CPU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_visualize", action="store_true", help="Disable visualization")
    parser.add_argument("--no_save", action="store_true", help="Disable model saving")
    
    args = parser.parse_args()
    
    config = MainConfig()
    config.preprocessed_dir = args.preprocessed_dir
    config.output_dir = args.output_dir
    config.device = args.device
    config.seed = args.seed
    config.visualize = not args.no_visualize
    config.save_model = not args.no_save
    
    config.model.encoder_type = args.encoder_type
    config.model.hidden_channels = args.hidden_channels
    config.model.out_channels = args.out_channels
    config.model.num_layers = args.num_layers
    config.model.num_heads = args.num_heads
    config.model.mask_rate = args.mask_rate
    config.model.dropout = args.dropout
    
    config.train.epochs = args.epochs
    config.train.lr = args.lr
    config.train.patience = args.patience
    
    run_main(config)


if __name__ == "__main__":
    main()
