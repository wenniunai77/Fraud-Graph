"""
类别 Embedding 预训练模块
使用无监督任务（Masked Attribute Modeling）训练类别 embedding
"""
import logging
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from configs.embedding_config import EmbeddingPretrainConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


class MaskedAttributePredictor(nn.Module):
    """
    Masked Attribute Modeling (MAM) 预训练模型
    
    思路：随机 mask 掉某些类别字段，用其他字段预测被 mask 的字段
    类似于 BERT 的 MLM，但用于类别特征
    """
    
    def __init__(
        self,
        category_sizes: Dict[str, int],  # {field_name: num_categories}
        embedding_dim: int = 8,
        hidden_dim: int = 64,
        embedding_dims: Optional[Dict[str, int]] = None  # 自适应维度: {field_name: dim}
    ):
        super().__init__()
        
        self.category_sizes = category_sizes
        self.embedding_dim = embedding_dim
        self.field_names = list(category_sizes.keys())
        
        # 如果提供了自适应维度，使用它；否则使用统一维度
        if embedding_dims is None:
            embedding_dims = {field: embedding_dim for field in self.field_names}
        self.embedding_dims = embedding_dims
        
        # 为每个类别字段创建 embedding（可能有不同维度）
        self.embeddings = nn.ModuleDict({
            field: nn.Embedding(num_cat, embedding_dims[field])
            for field, num_cat in category_sizes.items()
        })
        
        # 初始化 embedding
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        
        # 预测头：用所有字段的 embedding 拼接后预测被 mask 的字段
        total_emb_dim = sum(embedding_dims.values())
        
        self.predictor = nn.Sequential(
            nn.Linear(total_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 为每个字段创建分类头
        self.classifiers = nn.ModuleDict({
            field: nn.Linear(hidden_dim, num_cat)
            for field, num_cat in category_sizes.items()
        })
    
    def forward(
        self,
        field_indices: Dict[str, torch.Tensor],  # {field_name: [batch_size]}
        mask_info: Optional[Dict[str, torch.Tensor]] = None  # {field_name: [batch_size] bool mask}
    ):
        """
        前向传播
        
        Args:
            field_indices: 每个字段的类别索引（batch_size,）
            mask_info: 每个字段是否被 mask（True 表示被 mask）
        
        Returns:
            如果提供 mask_info，返回预测 logits；否则返回 embeddings
        """
        batch_size = next(iter(field_indices.values())).shape[0]
        
        # 获取每个字段的 embedding
        field_embeddings = {}
        for field, indices in field_indices.items():
            emb = self.embeddings[field](indices)  # [batch_size, emb_dim]
            
            # 如果被 mask，用零向量替代
            if mask_info is not None and field in mask_info:
                mask = mask_info[field].unsqueeze(-1)  # [batch_size, 1]
                emb = emb * (~mask).float()  # mask=True 的位置置零
            
            field_embeddings[field] = emb
        
        # 拼接所有字段的 embedding
        concat_emb = torch.cat([field_embeddings[f] for f in self.field_names], dim=-1)
        
        # 如果不是训练模式，直接返回 embedding
        if mask_info is None:
            return field_embeddings
        
        # 通过预测头
        hidden = self.predictor(concat_emb)  # [batch_size, hidden_dim]
        
        # 预测每个被 mask 的字段
        predictions = {
            field: self.classifiers[field](hidden)
            for field in self.field_names
        }
        
        return predictions
    
    def compute_loss(
        self,
        field_indices: Dict[str, torch.Tensor],
        mask_info: Dict[str, torch.Tensor]
    ):
        """
        计算 Masked Attribute Modeling 损失
        
        只对被 mask 的字段计算交叉熵损失
        """
        predictions = self.forward(field_indices, mask_info)
        
        total_loss = 0.0
        total_count = 0
        field_losses = {}
        
        for field in self.field_names:
            mask = mask_info[field]  # [batch_size]
            
            if mask.sum() > 0:  # 如果有被 mask 的样本
                pred_logits = predictions[field][mask]  # [num_masked, num_classes]
                true_labels = field_indices[field][mask]  # [num_masked]
                
                loss = F.cross_entropy(pred_logits, true_labels)
                total_loss += loss * mask.sum().item()
                total_count += mask.sum().item()
                field_losses[field] = loss.item()
        
        if total_count > 0:
            avg_loss = total_loss / total_count
        else:
            avg_loss = torch.tensor(0.0, device=next(iter(field_indices.values())).device)
        
        return avg_loss, field_losses


class EmbeddingPretrainer:
    """Embedding 预训练器"""
    
    def __init__(self, config: EmbeddingPretrainConfig):
        self.config = config
        self.model: Optional[MaskedAttributePredictor] = None
        self.optimizer = None
        self.scheduler = None
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"使用设备: {self.device}")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        categorical_cols: List[int],
        col_name_map: Dict[int, str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, int]]]:
        """
        准备训练数据
        
        P2 修复: 排序 unique 确保映射稳定（跨次运行一致）
        
        Returns:
            - field_data: {field_name: [num_samples] 类别索引}
            - category_mappings: {field_name: {category_value: index}}
        """
        field_data = {}
        category_mappings = {}
        
        for col_i in categorical_cols:
            # P2 修复: 先 fillna 再 astype(str)，避免 NaN 变成字符串 "nan"
            col_data = df.iloc[:, col_i].fillna('UNKNOWN').astype(str)
            field_name = col_name_map.get(col_i, f"col_{col_i}")
            
            # 构建类别映射（排序确保稳定性）
            unique_values = sorted(col_data.unique())  # P2: 排序确保稳定
            cat_to_idx = {val: idx for idx, val in enumerate(unique_values)}
            
            # 转换为索引
            indices = np.array([cat_to_idx[val] for val in col_data], dtype=np.int64)
            
            field_data[field_name] = indices
            category_mappings[field_name] = cat_to_idx
        
        return field_data, category_mappings
    
    def create_masked_batch(
        self,
        batch_indices: np.ndarray,
        field_data: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        创建一个 batch 的数据并进行 mask
        
        Returns:
            - field_indices: {field_name: [batch_size] 原始索引}
            - mask_info: {field_name: [batch_size] bool，True 表示被 mask}
        """
        batch_size = len(batch_indices)
        
        field_indices = {}
        mask_info = {}
        
        for field, data in field_data.items():
            indices = torch.tensor(data[batch_indices], dtype=torch.long, device=self.device)
            field_indices[field] = indices
            
            # 随机 mask
            mask = torch.rand(batch_size, device=self.device) < self.config.mask_prob
            mask_info[field] = mask
        
        return field_indices, mask_info
    
    def train(
        self,
        df: pd.DataFrame,
        categorical_cols: List[int],
        col_name_map: Dict[int, str],
        use_adaptive_dim: bool = True,
        dim_multiplier: float = 0.25,
        max_dim: int = 32,
        min_dim: int = 4
    ):
        """训练 embedding
        
        Args:
            df: 数据框
            categorical_cols: 类别特征列索引
            col_name_map: 列索引到名称的映射
            use_adaptive_dim: 是否使用自适应维度
            dim_multiplier: 维度计算的幂次（默认 0.25）
            max_dim: 最大维度
            min_dim: 最小维度
        """
        logging.info("=" * 60)
        logging.info("开始 Embedding 预训练（Masked Attribute Modeling）")
        logging.info("=" * 60)
        
        # 准备数据
        logging.info("准备训练数据...")
        field_data, category_mappings = self.prepare_data(df, categorical_cols, col_name_map)
        
        num_samples = len(df)
        logging.info(f"总样本数: {num_samples:,}")
        logging.info(f"类别字段: {list(field_data.keys())}")
        
        # 计算每个字段的 embedding 维度
        category_sizes = {field: len(mapping) for field, mapping in category_mappings.items()}
        
        if use_adaptive_dim:
            logging.info("使用自适应 embedding 维度:")
            embedding_dims = {}
            for field, num_cat in category_sizes.items():
                calculated_dim = int(num_cat ** dim_multiplier)
                dim = min(max_dim, max(min_dim, calculated_dim))
                embedding_dims[field] = dim
                logging.info(f"  - {field}: {num_cat} 类别 -> {dim} 维")
        else:
            logging.info(f"使用固定 embedding 维度: {self.config.embedding_dim}")
            embedding_dims = {field: self.config.embedding_dim for field in category_sizes.keys()}
        
        # 划分训练/验证集
        val_size = int(num_samples * self.config.validation_split)
        train_size = num_samples - val_size
        indices = np.random.permutation(num_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        logging.info(f"训练集: {train_size:,}, 验证集: {val_size:,}")
        
        # 创建模型（使用自适应或固定维度）
        avg_emb_dim = int(np.mean(list(embedding_dims.values())))
        self.model = MaskedAttributePredictor(
            category_sizes=category_sizes,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=avg_emb_dim * 4,
            embedding_dims=embedding_dims  # 传入自适应维度
        ).to(self.device)
        
        logging.info(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 学习率调度器
        if self.config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # 训练阶段
            train_loss = self._train_epoch(epoch, train_indices, field_data)
            
            # 验证阶段
            val_loss = self._validate_epoch(val_indices, field_data)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
            )
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_embeddings(category_mappings)
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logging.info("=" * 60)
        logging.info(f"预训练完成！最佳验证损失: {best_val_loss:.4f}")
        logging.info(f"Embedding 权重已保存: {self.config.save_path}")
        logging.info("=" * 60)
    
    def _train_epoch(
        self,
        epoch: int,
        train_indices: np.ndarray,
        field_data: Dict[str, np.ndarray]
    ) -> float:
        """训练一个 epoch"""
        self.model.train()
        
        batch_size = self.config.batch_size
        num_batches = (len(train_indices) + batch_size - 1) // batch_size
        
        total_loss = 0.0
        
        # 打乱训练数据
        shuffled_indices = np.random.permutation(train_indices)
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1} Training")
        
        for i in pbar:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(shuffled_indices))
            batch_indices = shuffled_indices[start_idx:end_idx]
            
            # 创建 masked batch
            field_indices, mask_info = self.create_masked_batch(batch_indices, field_data)
            
            # 前向传播
            self.optimizer.zero_grad()
            loss, _ = self.model.compute_loss(field_indices, mask_info)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (i + 1) % self.config.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / num_batches
    
    def _validate_epoch(
        self,
        val_indices: np.ndarray,
        field_data: Dict[str, np.ndarray]
    ) -> float:
        """验证一个 epoch"""
        self.model.eval()
        
        batch_size = self.config.batch_size
        num_batches = (len(val_indices) + batch_size - 1) // batch_size
        
        total_loss = 0.0
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(val_indices))
                batch_indices = val_indices[start_idx:end_idx]
                
                field_indices, mask_info = self.create_masked_batch(batch_indices, field_data)
                loss, _ = self.model.compute_loss(field_indices, mask_info)
                
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def _save_embeddings(self, category_mappings: Dict[str, Dict[str, int]]):
        """保存预训练的 embedding 权重"""
        os.makedirs(os.path.dirname(self.config.save_path), exist_ok=True)
        
        save_dict = {
            "embeddings": {
                field: emb.weight.data.cpu()
                for field, emb in self.model.embeddings.items()
            },
            "category_mappings": category_mappings,
            "embedding_dims": self.model.embedding_dims,  # 保存每个字段的维度
            "config": {
                "embedding_dim": self.config.embedding_dim,
                "pretrain_method": self.config.pretrain_method
            }
        }
        
        torch.save(save_dict, self.config.save_path)
        logging.info(f"✓ Embedding 权重已保存: {self.config.save_path}")


def load_pretrained_embeddings(path: str) -> Dict:
    """
    加载预训练的 embedding 权重
    
    Returns:
        包含 embeddings, category_mappings, config 的字典
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"预训练 embedding 文件不存在: {path}")
    
    return torch.load(path, weights_only=False)
