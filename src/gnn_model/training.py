"""
Training module for Graph Neural Network fraud detection model.

This module provides a comprehensive training pipeline with MLflow integration
for experiment tracking, model checkpointing, and performance monitoring.
It implements early stopping, learning rate scheduling, and comprehensive
evaluation metrics for the fraud detection task.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import dgl
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from .model import GraphSAGEClassifier, GraphDataLoader
from ..config import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting during training.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        restore_best_weights: Whether to restore best weights when stopping
    """
    
    def __init__(
        self,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            score: Current validation score (higher is better)
            model: Model to potentially save weights from
            
        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    self.restore_checkpoint(model)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
            
        return False
    
    def save_checkpoint(self, model: nn.Module) -> None:
        """Save the current best model weights."""
        if self.restore_best_weights:
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    def restore_checkpoint(self, model: nn.Module) -> None:
        """Restore the best model weights."""
        if self.best_weights is not None:
            model.load_state_dict({k: v.to(model.device) for k, v in self.best_weights.items()})


class MetricsCalculator:
    """
    Utility class for calculating comprehensive evaluation metrics.
    """
    
    @staticmethod
    def calculate_binary_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate comprehensive binary classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            threshold: Decision threshold for binary classification
            
        Returns:
            Dict[str, float]: Dictionary of calculated metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        # Probability-based metrics (if probabilities are provided)
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        
        return metrics
    
    @staticmethod
    def create_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> str:
        """
        Create a detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            str: Formatted classification report
        """
        metrics = MetricsCalculator.calculate_binary_metrics(y_true, y_pred, y_prob)
        
        report = f"""
Classification Report:
====================
Accuracy:           {metrics['accuracy']:.4f}
Precision:          {metrics['precision']:.4f}
Recall:             {metrics['recall']:.4f}
F1-Score:           {metrics['f1_score']:.4f}
Specificity:        {metrics['specificity']:.4f}
Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}

Confusion Matrix:
                 Predicted
Actual     0         1
0      {metrics['true_negatives']:6d}   {metrics['false_positives']:6d}
1      {metrics['false_negatives']:6d}   {metrics['true_positives']:6d}
"""
        
        if y_prob is not None:
            report += f"""
ROC AUC:            {metrics['roc_auc']:.4f}
PR AUC:             {metrics['pr_auc']:.4f}
"""
        
        return report


class FraudDetectionTrainer:
    """
    Comprehensive trainer for Graph Neural Network fraud detection model.
    
    This class provides a complete training pipeline with MLflow integration,
    early stopping, learning rate scheduling, and comprehensive evaluation.
    """
    
    def __init__(
        self,
        model: GraphSAGEClassifier,
        device: str = 'cpu',
        experiment_name: str = config.MLFLOW_EXPERIMENT_NAME
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            model: GraphSAGE classifier model
            device: Device to train on ('cpu' or 'cuda')
            experiment_name: MLflow experiment name
        """
        self.model = model.to(device)
        self.device = device
        self.experiment_name = experiment_name
        
        # Training components
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[nn.Module] = None
        self.early_stopping: Optional[EarlyStopping] = None
        
        # Training history
        self.train_history: List[Dict[str, float]] = []
        self.val_history: List[Dict[str, float]] = []
        self.best_val_score: float = 0.0
        self.best_epoch: int = 0
        
        # MLflow setup
        self._setup_mlflow()
        
        logger.info(f"Trainer initialized for device: {device}")
        
    def _setup_mlflow(self) -> None:
        """Setup MLflow experiment and tracking."""
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created MLflow experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"Using MLflow experiment: {self.experiment_name}")
            
        except Exception as e:
            logger.warning(f"MLflow setup failed: {str(e)}. Continuing without MLflow.")
    
    def setup_training(
        self,
        learning_rate: float = config.LEARNING_RATE,
        weight_decay: float = 1e-5,
        optimizer_type: str = 'adam',
        scheduler_type: str = 'plateau',
        class_weights: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None
    ) -> None:
        """
        Setup training components (optimizer, scheduler, criterion).
        
        Args:
            learning_rate: Learning rate for optimization
            weight_decay: L2 regularization weight
            optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
            scheduler_type: Type of LR scheduler ('plateau', 'cosine', 'step')
            class_weights: Weights for different classes
            pos_weight: Positive class weight for BCEWithLogitsLoss
        """
        # Setup optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Setup learning rate scheduler
        if scheduler_type.lower() == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_type.lower() == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.NUM_EPOCHS,
                eta_min=learning_rate * 0.01
            )
        
        # Setup loss criterion
        if pos_weight is not None:
            pos_weight = pos_weight.to(self.device)
        
        self.criterion = nn.BCEWithLogitsLoss(
            weight=class_weights,
            pos_weight=pos_weight
        )
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )
        
        logger.info(f"Training setup completed:")
        logger.info(f"  Optimizer: {optimizer_type} (LR: {learning_rate})")
        logger.info(f"  Scheduler: {scheduler_type}")
        logger.info(f"  Criterion: BCEWithLogitsLoss")
    
    def train_epoch(
        self,
        graph: dgl.DGLGraph,
        train_mask: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            graph: Training graph
            train_mask: Mask for training nodes
            edge_weight: Optional edge weights
            
        Returns:
            Dict[str, float]: Training metrics for the epoch
        """
        self.model.train()
        
        # Forward pass
        logits = self.model(graph, graph.ndata['feat'], edge_weight)
        
        # Get training predictions and labels
        train_logits = logits[train_mask]
        train_labels = graph.ndata['label'][train_mask]
        
        # Calculate loss
        loss = self.criterion(train_logits.squeeze(), train_labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Calculate training metrics
        with torch.no_grad():
            train_probs = torch.sigmoid(train_logits.squeeze()).cpu().numpy()
            train_preds = (train_probs > 0.5).astype(int)
            train_labels_np = train_labels.cpu().numpy()
            
            metrics = MetricsCalculator.calculate_binary_metrics(
                train_labels_np,
                train_preds,
                train_probs
            )
            metrics['loss'] = loss.item()
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        graph: dgl.DGLGraph,
        eval_mask: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        prefix: str = 'val'
    ) -> Dict[str, float]:
        """
        Evaluate the model on a given set.
        
        Args:
            graph: Evaluation graph
            eval_mask: Mask for evaluation nodes
            edge_weight: Optional edge weights
            prefix: Prefix for metric names
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        self.model.eval()
        
        # Forward pass
        logits = self.model(graph, graph.ndata['feat'], edge_weight)
        
        # Get evaluation predictions and labels
        eval_logits = logits[eval_mask]
        eval_labels = graph.ndata['label'][eval_mask]
        
        # Calculate loss
        loss = self.criterion(eval_logits.squeeze(), eval_labels)
        
        # Calculate metrics
        eval_probs = torch.sigmoid(eval_logits.squeeze()).cpu().numpy()
        eval_preds = (eval_probs > 0.5).astype(int)
        eval_labels_np = eval_labels.cpu().numpy()
        
        metrics = MetricsCalculator.calculate_binary_metrics(
            eval_labels_np,
            eval_preds,
            eval_probs
        )
        metrics['loss'] = loss.item()
        
        # Add prefix to metric names
        prefixed_metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
        
        return prefixed_metrics
    
    def train(
        self,
        graph: dgl.DGLGraph,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        num_epochs: int = config.NUM_EPOCHS,
        edge_weight: Optional[torch.Tensor] = None,
        log_every: int = 10,
        save_every: int = config.SAVE_EVERY_N_EPOCHS
    ) -> Dict[str, Any]:
        """
        Complete training loop with validation and early stopping.
        
        Args:
            graph: Training graph
            train_mask: Training node mask
            val_mask: Validation node mask
            num_epochs: Number of training epochs
            edge_weight: Optional edge weights
            log_every: Log metrics every N epochs
            save_every: Save checkpoint every N epochs
            
        Returns:
            Dict[str, Any]: Training summary
        """
        # Start MLflow run
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")
            
            # Log hyperparameters
            self._log_hyperparameters()
            
            # Training loop
            start_time = time.time()
            
            for epoch in range(num_epochs):
                epoch_start = time.time()
                
                # Train for one epoch
                train_metrics = self.train_epoch(graph, train_mask, edge_weight)
                
                # Validate
                val_metrics = self.evaluate(graph, val_mask, edge_weight, 'val')
                
                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_f1_score'])
                    else:
                        self.scheduler.step()
                
                # Store history
                train_metrics['epoch'] = epoch
                val_metrics['epoch'] = epoch
                train_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
                
                self.train_history.append(train_metrics)
                self.val_history.append(val_metrics)
                
                # Update best validation score
                current_val_score = val_metrics['val_f1_score']
                if current_val_score > self.best_val_score:
                    self.best_val_score = current_val_score
                    self.best_epoch = epoch
                
                # Log metrics to MLflow
                self._log_epoch_metrics(epoch, train_metrics, val_metrics)
                
                # Print progress
                if epoch % log_every == 0 or epoch == num_epochs - 1:
                    epoch_time = time.time() - epoch_start
                    logger.info(
                        f"Epoch {epoch:3d}/{num_epochs} ({epoch_time:.2f}s) | "
                        f"Train Loss: {train_metrics['loss']:.4f} | "
                        f"Train F1: {train_metrics['f1_score']:.4f} | "
                        f"Val Loss: {val_metrics['val_loss']:.4f} | "
                        f"Val F1: {val_metrics['val_f1_score']:.4f} | "
                        f"LR: {train_metrics['learning_rate']:.6f}"
                    )
                
                # Save checkpoint
                if save_every > 0 and (epoch + 1) % save_every == 0:
                    self._save_checkpoint(epoch, run_id)
                
                # Check early stopping
                if self.early_stopping(current_val_score, self.model):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    logger.info(f"Best validation F1: {self.best_val_score:.4f} at epoch {self.best_epoch}")
                    break
            
            # Training completed
            total_time = time.time() - start_time
            
            # Log final model
            self._log_final_model(run_id)
            
            # Create training summary
            training_summary = {
                'run_id': run_id,
                'best_epoch': self.best_epoch,
                'best_val_f1': self.best_val_score,
                'total_epochs': len(self.train_history),
                'total_time': total_time,
                'final_lr': self.optimizer.param_groups[0]['lr']
            }
            
            logger.info(f"Training completed in {total_time:.2f}s")
            logger.info(f"Best validation F1: {self.best_val_score:.4f} at epoch {self.best_epoch}")
            
            return training_summary
    
    def _log_hyperparameters(self) -> None:
        """Log hyperparameters to MLflow."""
        params = {
            'model_type': 'GraphSAGEClassifier',
            'learning_rate': config.LEARNING_RATE,
            'batch_size': config.BATCH_SIZE,
            'num_epochs': config.NUM_EPOCHS,
            'early_stopping_patience': config.EARLY_STOPPING_PATIENCE,
            'gnn_input_dim': config.GNN_INPUT_DIM,
            'gnn_hidden_dim': config.GNN_HIDDEN_DIM,
            'gnn_output_dim': config.GNN_OUTPUT_DIM,
            'gnn_num_layers': config.GNN_NUM_LAYERS,
            'gnn_dropout_rate': config.GNN_DROPOUT_RATE,
            'classifier_hidden_dim': config.CLASSIFIER_HIDDEN_DIM,
        }
        
        # Add model-specific parameters
        model_summary = self.model.get_model_summary()
        params.update({
            'total_parameters': model_summary['total_parameters'],
            'aggregator_type': model_summary['architecture']['aggregator_type']
        })
        
        mlflow.log_params(params)
    
    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """Log metrics for a single epoch to MLflow."""
        # Combine all metrics
        all_metrics = {**train_metrics, **val_metrics}
        
        # Log to MLflow
        for metric_name, metric_value in all_metrics.items():
            if metric_name != 'epoch':  # Don't log epoch as a metric
                mlflow.log_metric(metric_name, metric_value, step=epoch)
    
    def _save_checkpoint(self, epoch: int, run_id: str) -> None:
        """Save model checkpoint."""
        checkpoint_dir = config.CHECKPOINTS_DIR / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_score': self.best_val_score,
            'train_history': self.train_history,
            'val_history': self.val_history
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def _log_final_model(self, run_id: str) -> None:
        """Log the final trained model to MLflow."""
        try:
            # Log the model
            mlflow.pytorch.log_model(
                self.model,
                "model",
                registered_model_name=config.MLFLOW_MODEL_NAME
            )
            
            # Log model summary
            model_summary = self.model.get_model_summary()
            mlflow.log_dict(model_summary, "model_summary.json")
            
            # Log training history
            train_df = pd.DataFrame(self.train_history)
            val_df = pd.DataFrame(self.val_history)
            
            train_df.to_csv("train_history.csv", index=False)
            val_df.to_csv("val_history.csv", index=False)
            
            mlflow.log_artifact("train_history.csv")
            mlflow.log_artifact("val_history.csv")
            
            # Clean up temporary files
            os.remove("train_history.csv")
            os.remove("val_history.csv")
            
            logger.info("Model and artifacts logged to MLflow successfully")
            
        except Exception as e:
            logger.warning(f"Failed to log model to MLflow: {str(e)}")
    
    def plot_training_history(self, save_path: Optional[Path] = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.train_history or not self.val_history:
            logger.warning("No training history available to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Convert history to DataFrames
        train_df = pd.DataFrame(self.train_history)
        val_df = pd.DataFrame(self.val_history)
        
        # Plot loss
        axes[0, 0].plot(train_df['epoch'], train_df['loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(val_df['epoch'], val_df['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot F1 score
        axes[0, 1].plot(train_df['epoch'], train_df['f1_score'], label='Train F1', color='blue')
        axes[0, 1].plot(val_df['epoch'], val_df['val_f1_score'], label='Val F1', color='red')
        axes[0, 1].set_title('F1 Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot ROC AUC
        axes[1, 0].plot(train_df['epoch'], train_df['roc_auc'], label='Train ROC AUC', color='blue')
        axes[1, 0].plot(val_df['epoch'], val_df['val_roc_auc'], label='Val ROC AUC', color='red')
        axes[1, 0].set_title('ROC AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('ROC AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot learning rate
        axes[1, 1].plot(train_df['epoch'], train_df['learning_rate'], label='Learning Rate', color='green')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
