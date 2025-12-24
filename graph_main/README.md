# GraphMAE Fraud Detection Project

A payment transaction fraud detection system based on GraphMAE (Graph Masked Autoencoder).

## Project Structure

The project is divided into two parts:

```
graph_main/
├── preprocess/                    # [Part 1] Data Preprocessing Module
│   ├── config.py                  # Preprocessing configuration
│   ├── data_loader.py             # Data loading
│   ├── feature_engineer.py        # Feature engineering (including time_diff)
│   ├── graph_builder.py           # Graph construction
│   ├── statistics.py              # Statistical analysis
│   ├── run_preprocess.py          # Preprocessing main script
│   ├── preprocess_check.ipynb     # Preprocessing check notebook
│   └── __init__.py                # Module exports
│
├── config.py                      # [Part 2] Main program configuration
├── models/                        # Model definitions
│   ├── graphmae.py               # GraphMAE model
│   ├── encoder.py                # Encoder (GAT/GCN)
│   ├── loss_func.py              # Loss function (SCE)
│   └── utils.py                  # Utility functions
├── trainer.py                     # Trainer
├── anomaly_detector.py           # Anomaly detector
├── visualization.py              # Visualization
├── run_main.py                   # Main program script
└── __init__.py                   # Module exports
```

## Running Process

### Must be run in order:

**Step 1: Data Preprocessing**
```bash
cd graph_main/preprocess
python run_preprocess.py \
    --data_path /path/to/your/data.csv \
    --output_dir ./preprocessed_data \
    --sample_size 0  # 0 means full dataset, or specify sample size e.g. 500000
```

**Step 2: Check Preprocessing Results (Optional but Recommended)**

Open `preprocess/preprocess_check.ipynb` in Jupyter and run all cells to check:
- fillna status
- time_diff feature construction success
- Data quality check

**Step 3: Run Main Program**
```bash
cd graph_main
python run_main.py \
    --preprocessed_dir ./preprocess/preprocessed_data \
    --output_dir ./output \
    --epochs 500 \
    --device 0
```

## Feature Description

### Edge Features

| Feature Name | Source Column | Description |
|--------------|---------------|-------------|
| instructed_amount | Column 6 | Customer specified amount |
| payment_amount | Column 8 | Bank used amount |
| credit_amount | Column 10 | Beneficiary received amount |
| payment_channel_encoded | Column 1 | Transaction channel encoding |
| debit_bic_code_encoded | Column 2 | Payer BIC code encoding |
| bene_bic_code_encoded | Column 3 | Beneficiary BIC code encoding |
| instructed_currency_encoded | Column 5 | Customer specified currency encoding |
| payment_currency_encoded | Column 7 | Bank used currency encoding |
| credit_currency_encoded | Column 9 | Beneficiary received currency encoding |
| mop_encoded | Column 13 | Method of payment encoding |
| txn_hour | Column 11 | Transaction hour |
| txn_day_of_week | Column 11 | Transaction day of week |
| txn_day_of_month | Column 11 | Transaction day of month |
| tds_hour | Column 12 | Storage hour |
| tds_day_of_week | Column 12 | Storage day of week |
| tds_day_of_month | Column 12 | Storage day of month |
| **time_diff_seconds** | Column 12-11 | **Time difference (tds_dt - txn_dt)** |

### Node Features

| Feature Name | Description |
|--------------|-------------|
| src_avg_feat_* | Average of edge features when acting as sender |
| dst_avg_feat_* | Average of edge features when acting as receiver |
| src_tx_count | Out-degree: number of transactions as sender |
| dst_tx_count | In-degree: number of transactions as receiver |
| total_degree | Total degree |
| in_out_ratio | In/out degree ratio |

## Output Files

### Preprocessing Output (`preprocess/preprocessed_data/`)
- `graph_data.pt`: PyG graph data object
- `node_features.pt`: Node features
- `edge_features.pt`: Edge features
- `edge_index.pt`: Edge index
- `node_mapping.pkl`: Node mapping
- `statistics.json`: Statistics information
- `preprocess_meta.json`: Preprocessing metadata (for checking)

### Main Program Output (`output/`)
- `graphmae_model.pt`: Trained model
- `anomaly_results.json`: Anomaly detection results
- `comprehensive_report.png`: Comprehensive visualization report
- `embeddings_tsne.png`: t-SNE visualization

## Dependencies Installation

```bash
pip install torch torch_geometric torch_scatter
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Key Parameters Description

### Preprocessing Parameters
- `--sample_size`: Sample size, 0 means full dataset
- `--src_col`: Source node column index (default 14, payer account)
- `--dst_col`: Target node column index (default 15, beneficiary account)

### Model Parameters
- `--encoder_type`: Encoder type (gat/gcn)
- `--hidden_channels`: Hidden layer dimension
- `--num_layers`: Number of GNN layers
- `--mask_rate`: Mask ratio
- `--epochs`: Training epochs
- `--patience`: Early stopping patience

## Checkpoint

After preprocessing, be sure to run `preprocess_check.ipynb` to verify:

1. ✅ fillna status: which columns were filled with missing values
2. ✅ time_diff construction: whether time difference feature was calculated successfully
3. ✅ Data quality: whether there are NaN, Inf or other abnormal values
4. ✅ Feature distribution: whether feature distribution is reasonable
