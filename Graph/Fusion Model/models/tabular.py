"""
表格无监督异常检测模型
包括：Isolation Forest、LOF、AutoEncoder
"""
import logging
import numpy as np
from typing import Dict, Optional, Tuple, List
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from configs import TabularModelConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)


class AutoEncoder(nn.Module):
    """AutoEncoder 用于异常检测"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32, 16, 32, 64],
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def compute_reconstruction_error(self, x):
        """计算重构误差"""
        with torch.no_grad():
            recon = self.forward(x)
            error = torch.mean((x - recon) ** 2, dim=1)
        return error


class TabularAnomalyDetector:
    """表格无监督异常检测器"""
    
    def __init__(self, config: TabularModelConfig):
        self.config = config
        self.scaler = StandardScaler()
        
        self.isolation_forest = None
        self.lof = None
        self.autoencoder = None
        
        self._fitted = False
    
    def fit(self, X: np.ndarray, device: str = "cpu"):
        """训练模型"""
        logging.info(f"训练表格异常检测模型 (type={self.config.model_type})...")
        logging.info(f"数据维度: {X.shape}")
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        model_type = self.config.model_type
        
        if model_type in ["isolation_forest", "ensemble"]:
            self._fit_isolation_forest(X_scaled)
        
        if model_type in ["lof", "ensemble"]:
            self._fit_lof(X_scaled)
        
        if model_type in ["autoencoder", "ensemble"]:
            self._fit_autoencoder(X_scaled, device)
        
        self._fitted = True
        logging.info("表格异常检测模型训练完成")
    
    def _fit_isolation_forest(self, X: np.ndarray):
        """训练 Isolation Forest"""
        logging.info("训练 Isolation Forest...")
        
        self.isolation_forest = IsolationForest(
            n_estimators=self.config.if_n_estimators,
            contamination=self.config.if_contamination,
            max_samples=self.config.if_max_samples,
            random_state=self.config.if_random_state,
            n_jobs=-1
        )
        self.isolation_forest.fit(X)
        logging.info("Isolation Forest 训练完成")
    
    def _fit_lof(self, X: np.ndarray):
        """训练 LOF"""
        logging.info("训练 LOF...")
        
        self.lof = LocalOutlierFactor(
            n_neighbors=self.config.lof_n_neighbors,
            contamination=self.config.lof_contamination,
            metric=self.config.lof_metric,
            novelty=True,  # 允许对新数据预测
            n_jobs=-1
        )
        self.lof.fit(X)
        logging.info("LOF 训练完成")
    
    def _fit_autoencoder(self, X: np.ndarray, device: str = "cpu"):
        """训练 AutoEncoder"""
        logging.info("训练 AutoEncoder...")
        
        input_dim = X.shape[1]
        self.autoencoder = AutoEncoder(
            input_dim=input_dim,
            hidden_dims=self.config.ae_hidden_dims,
            dropout=self.config.ae_dropout
        ).to(device)
        
        # 转为 tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        
        optimizer = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=self.config.ae_lr
        )
        criterion = nn.MSELoss()
        
        batch_size = self.config.ae_batch_size
        n_samples = X.shape[0]
        
        self.autoencoder.train()
        for epoch in range(self.config.ae_epochs):
            total_loss = 0
            n_batches = 0
            
            # 随机打乱
            perm = torch.randperm(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch = X_tensor[batch_idx]
                
                optimizer.zero_grad()
                recon = self.autoencoder(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / n_batches
                logging.info(f"  Epoch {epoch+1}/{self.config.ae_epochs}, Loss: {avg_loss:.6f}")
        
        self.autoencoder.eval()
        logging.info("AutoEncoder 训练完成")
    
    def predict_scores(self, X: np.ndarray, device: str = "cpu") -> Dict[str, np.ndarray]:
        """预测异常分数"""
        if not self._fitted:
            raise RuntimeError("模型未训练！请先调用 fit()")
        
        X_scaled = self.scaler.transform(X)
        
        scores = {}
        model_type = self.config.model_type
        
        if model_type in ["isolation_forest", "ensemble"]:
            # Isolation Forest: 返回负的 decision function（越大越异常）
            if_scores = -self.isolation_forest.decision_function(X_scaled)
            scores["isolation_forest"] = self._normalize_scores(if_scores)
        
        if model_type in ["lof", "ensemble"]:
            # LOF: 返回负的 decision function（越大越异常）
            lof_scores = -self.lof.decision_function(X_scaled)
            scores["lof"] = self._normalize_scores(lof_scores)
        
        if model_type in ["autoencoder", "ensemble"]:
            # AutoEncoder: 返回重构误差（越大越异常）
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            ae_scores = self.autoencoder.compute_reconstruction_error(X_tensor).cpu().numpy()
            scores["autoencoder"] = self._normalize_scores(ae_scores)
        
        return scores
    
    def predict_fusion_score(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        """预测融合后的表格异常分数"""
        scores = self.predict_scores(X, device)
        
        if self.config.model_type == "ensemble":
            # 动态加权融合：只对存在的模型归一化权重
            raw_weights = self.config.ensemble_weights  # [IF, LOF, AE]
            model_keys = ["isolation_forest", "lof", "autoencoder"]
            
            # 收集存在的模型分数和对应权重
            active_scores = []
            active_weights = []
            for i, key in enumerate(model_keys):
                if key in scores:
                    active_scores.append(scores[key])
                    active_weights.append(raw_weights[i])
            
            if len(active_scores) == 0:
                logging.warning("没有可用的子模型分数，返回零分数")
                return np.zeros(X.shape[0])
            
            # 归一化权重使其和为 1
            weight_sum = sum(active_weights)
            normalized_weights = [w / weight_sum for w in active_weights]
            
            # 加权融合
            fusion_score = np.zeros_like(active_scores[0])
            for s, w in zip(active_scores, normalized_weights):
                fusion_score += s * w
            
            return self._normalize_scores(fusion_score)
        else:
            # 单模型
            return list(scores.values())[0]
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """归一化分数到 [0, 1]"""
        min_s = scores.min()
        max_s = scores.max()
        if max_s - min_s > 1e-8:
            return (scores - min_s) / (max_s - min_s)
        return np.zeros_like(scores)
    
    def get_training_info(self) -> Dict:
        """获取训练信息"""
        return {
            "model_type": self.config.model_type,
            "fitted": self._fitted,
            "has_isolation_forest": self.isolation_forest is not None,
            "has_lof": self.lof is not None,
            "has_autoencoder": self.autoencoder is not None,
            "ensemble_weights": self.config.ensemble_weights if self.config.model_type == "ensemble" else None,
        }
    
    def save(self, path: str):
        """保存模型"""
        import pickle
        import os
        
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        
        state = {
            "config": self.config,
            "scaler": self.scaler,
            "isolation_forest": self.isolation_forest,
            "lof": self.lof,
            "autoencoder_state": self.autoencoder.state_dict() if self.autoencoder else None,
            "autoencoder_input_dim": self.autoencoder.network[0].in_features if self.autoencoder else None,
            "_fitted": self._fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logging.info(f"表格模型已保存: {path}")
    
    def load(self, path: str, device: str = "cpu"):
        """加载模型"""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state["config"]
        self.scaler = state["scaler"]
        self.isolation_forest = state["isolation_forest"]
        self.lof = state["lof"]
        self._fitted = state["_fitted"]
        
        if state["autoencoder_state"] is not None:
            input_dim = state["autoencoder_input_dim"]
            self.autoencoder = AutoEncoder(
                input_dim=input_dim,
                hidden_dims=self.config.ae_hidden_dims,
                dropout=self.config.ae_dropout
            ).to(device)
            self.autoencoder.load_state_dict(state["autoencoder_state"])
            self.autoencoder.eval()
        
        logging.info(f"表格模型已加载: {path}")
