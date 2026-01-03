"""
Prediction module for Graph Neural Network fraud detection model.

This module provides functionality for loading trained models and making
fraud predictions on new transaction data. It includes utilities for
subgraph construction, batch prediction, and model interpretation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import pickle

import torch
import torch.nn.functional as F
import dgl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from .model import GraphSAGEClassifier, GraphDataLoader
from ..config import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class FraudPredictor:
    """
    Fraud prediction service using trained GraphSAGE model.
    
    This class handles loading trained models from MLflow, preprocessing
    new transaction data, and making fraud predictions with confidence scores.
    """
    
    def __init__(
        self,
        model_name: str = config.MLFLOW_MODEL_NAME,
        model_stage: str = config.MLFLOW_MODEL_STAGE,
        device: str = 'cpu'
    ) -> None:
        """
        Initialize the fraud predictor.
        
        Args:
            model_name: Name of the registered model in MLflow
            model_stage: Stage of the model to load ('Production', 'Staging', etc.)
            device: Device to run predictions on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.model_stage = model_stage
        self.device = device
        
        # Model components
        self.model: Optional[GraphSAGEClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.node_mapping: Optional[Dict[str, int]] = None
        
        # Prediction history
        self.prediction_history: List[Dict[str, Any]] = []
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        self.client = MlflowClient()
        
        logger.info(f"FraudPredictor initialized for device: {device}")
    
    def load_model(
        self,
        model_uri: Optional[str] = None,
        local_path: Optional[Path] = None
    ) -> None:
        """
        Load a trained model from MLflow or local path.
        
        Args:
            model_uri: MLflow model URI (e.g., 'models:/model_name/version')
            local_path: Local path to saved model
        """
        try:
            if model_uri:
                # Load from MLflow URI
                logger.info(f"Loading model from MLflow URI: {model_uri}")
                self.model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
                
            elif local_path:
                # Load from local path
                logger.info(f"Loading model from local path: {local_path}")
                checkpoint = torch.load(local_path, map_location=self.device)
                
                # Reconstruct model (you might need to adjust this based on saved format)
                model_config = checkpoint.get('model_config', {})
                self.model = GraphSAGEClassifier(**model_config)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
            else:
                # Load latest production model from MLflow
                model_uri = f"models:/{self.model_name}/{self.model_stage}"
                logger.info(f"Loading model from MLflow: {model_uri}")
                self.model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            logger.info(f"Model summary: {self.model.get_model_summary()}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def load_preprocessing_artifacts(
        self,
        scaler_path: Optional[Path] = None,
        feature_names_path: Optional[Path] = None,
        node_mapping_path: Optional[Path] = None
    ) -> None:
        """
        Load preprocessing artifacts (scaler, feature names, node mapping).
        
        Args:
            scaler_path: Path to saved StandardScaler
            feature_names_path: Path to saved feature names
            node_mapping_path: Path to saved node ID mapping
        """
        try:
            # Load scaler
            if scaler_path and scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded scaler from {scaler_path}")
            
            # Load feature names
            if feature_names_path and feature_names_path.exists():
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info(f"Loaded {len(self.feature_names)} feature names")
            
            # Load node mapping
            if node_mapping_path and node_mapping_path.exists():
                with open(node_mapping_path, 'r') as f:
                    self.node_mapping = json.load(f)
                logger.info(f"Loaded mapping for {len(self.node_mapping)} nodes")
                
        except Exception as e:
            logger.warning(f"Failed to load some preprocessing artifacts: {str(e)}")
    
    def preprocess_transaction(
        self,
        transaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Preprocess a single transaction for prediction.
        
        Args:
            transaction_data: Dictionary containing transaction details
            
        Returns:
            Dict[str, Any]: Preprocessed transaction data
        """
        processed = transaction_data.copy()
        
        # Add derived features
        processed['amount_log'] = np.log1p(processed.get('amount', 0))
        
        # Add time-based features
        step = processed.get('step', 0)
        processed['hour_of_day'] = step % 24
        processed['day_of_month'] = (step // 24) % 30
        
        # Encode transaction type
        type_mapping = {t: i for i, t in enumerate(config.TRANSACTION_TYPES)}
        processed['type_encoded'] = type_mapping.get(processed.get('type', ''), 0)
        
        return processed
    
    def create_subgraph_for_transaction(
        self,
        transaction_data: Dict[str, Any],
        full_graph: dgl.DGLGraph,
        num_hops: int = 2
    ) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        """
        Create a subgraph around the transaction participants.
        
        Args:
            transaction_data: Transaction details
            full_graph: Complete transaction graph
            num_hops: Number of hops for subgraph extraction
            
        Returns:
            Tuple of (subgraph, target_node_idx)
        """
        sender_id = transaction_data.get('sender_id')
        receiver_id = transaction_data.get('receiver_id')
        
        if not self.node_mapping:
            raise ValueError("Node mapping not loaded. Call load_preprocessing_artifacts() first.")
        
        # Get node indices
        sender_idx = self.node_mapping.get(sender_id)
        receiver_idx = self.node_mapping.get(receiver_id)
        
        if sender_idx is None or receiver_idx is None:
            raise ValueError(f"Unknown user IDs: {sender_id}, {receiver_id}")
        
        # Extract subgraph around both participants
        center_nodes = [sender_idx, receiver_idx]
        subgraph_nodes = dgl.khop_subgraph(full_graph, center_nodes, num_hops)[0]
        subgraph = full_graph.subgraph(subgraph_nodes)
        
        # Find sender index in subgraph
        sender_subgraph_idx = torch.where(subgraph_nodes == sender_idx)[0].item()
        
        return subgraph, torch.tensor(sender_subgraph_idx)
    
    def predict_fraud(
        self,
        transaction_data: Dict[str, Any],
        full_graph: Optional[dgl.DGLGraph] = None,
        return_confidence: bool = True,
        return_explanation: bool = False
    ) -> Dict[str, Any]:
        """
        Predict fraud probability for a single transaction.
        
        Args:
            transaction_data: Dictionary containing transaction details
                            Must include: sender_id, receiver_id, amount, type
            full_graph: Complete transaction graph (optional, for subgraph-based prediction)
            return_confidence: Whether to return prediction confidence
            return_explanation: Whether to return explanation features
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess transaction
            processed_transaction = self.preprocess_transaction(transaction_data)
            
            # Create prediction result
            result = {
                'transaction_id': processed_transaction.get('transaction_id', 'unknown'),
                'sender_id': processed_transaction.get('sender_id'),
                'receiver_id': processed_transaction.get('receiver_id'),
                'amount': processed_transaction.get('amount'),
                'type': processed_transaction.get('type')
            }
            
            if full_graph is not None:
                # Subgraph-based prediction
                subgraph, target_node_idx = self.create_subgraph_for_transaction(
                    processed_transaction, full_graph
                )
                
                # Make prediction
                with torch.no_grad():
                    logits = self.model(subgraph, subgraph.ndata['feat'])
                    target_logit = logits[target_node_idx]
                    fraud_probability = torch.sigmoid(target_logit).item()
                
                if return_explanation:
                    # Get attention weights for explainability
                    _, attention_dict = self.model.forward_with_attention(
                        subgraph, subgraph.ndata['feat']
                    )
                    
                    result['explanation_features'] = {
                        'node_embeddings': attention_dict['graph_embeddings'][target_node_idx].cpu().numpy().tolist(),
                        'subgraph_size': subgraph.num_nodes(),
                        'neighborhood_fraud_rate': subgraph.ndata.get('label', torch.zeros(subgraph.num_nodes())).mean().item()
                    }
            
            else:
                # Simple feature-based prediction (fallback)
                logger.warning("No graph provided, using simplified prediction")
                
                # Create simple feature vector from transaction
                features = np.array([
                    processed_transaction.get('amount_log', 0),
                    processed_transaction.get('type_encoded', 0),
                    processed_transaction.get('hour_of_day', 0),
                    processed_transaction.get('day_of_month', 0)
                ]).reshape(1, -1)
                
                if self.scaler:
                    features = self.scaler.transform(features)
                
                # Simple heuristic-based prediction (this would be enhanced in a real implementation)
                amount = processed_transaction.get('amount', 0)
                transaction_type = processed_transaction.get('type', '')
                
                # Basic fraud indicators
                high_amount_flag = amount > config.MAX_AMOUNT_THRESHOLD * 0.5
                suspicious_type_flag = transaction_type in ['CASH_OUT', 'TRANSFER']
                
                fraud_probability = 0.1  # Base probability
                if high_amount_flag:
                    fraud_probability += 0.3
                if suspicious_type_flag:
                    fraud_probability += 0.2
                
                fraud_probability = min(fraud_probability, 0.9)  # Cap at 90%
            
            # Add prediction results
            result['fraud_probability'] = fraud_probability
            result['is_fraud_predicted'] = fraud_probability > 0.5
            
            # Add confidence metrics
            if return_confidence:
                confidence = abs(fraud_probability - 0.5) * 2  # Distance from decision boundary
                result['confidence'] = confidence
                result['risk_level'] = self._get_risk_level(fraud_probability)
            
            # Store prediction in history
            self.prediction_history.append({
                'timestamp': pd.Timestamp.now().isoformat(),
                'transaction_id': result['transaction_id'],
                'fraud_probability': fraud_probability,
                'prediction': result['is_fraud_predicted']
            })
            
            logger.info(f"Prediction completed: {fraud_probability:.4f} for transaction {result['transaction_id']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_batch(
        self,
        transactions: List[Dict[str, Any]],
        full_graph: Optional[dgl.DGLGraph] = None,
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Predict fraud probabilities for a batch of transactions.
        
        Args:
            transactions: List of transaction dictionaries
            full_graph: Complete transaction graph
            batch_size: Number of transactions to process at once
            
        Returns:
            List[Dict[str, Any]]: Prediction results for all transactions
        """
        results = []
        
        logger.info(f"Processing batch of {len(transactions)} transactions")
        
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            
            for transaction in batch:
                try:
                    result = self.predict_fraud(
                        transaction,
                        full_graph=full_graph,
                        return_confidence=True
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed to process transaction {transaction.get('transaction_id', 'unknown')}: {str(e)}")
                    # Add error result
                    results.append({
                        'transaction_id': transaction.get('transaction_id', 'unknown'),
                        'error': str(e),
                        'fraud_probability': None,
                        'is_fraud_predicted': False
                    })
            
            logger.info(f"Processed batch {i // batch_size + 1}/{(len(transactions) + batch_size - 1) // batch_size}")
        
        return results
    
    def _get_risk_level(self, fraud_probability: float) -> str:
        """
        Convert fraud probability to risk level category.
        
        Args:
            fraud_probability: Fraud probability score
            
        Returns:
            str: Risk level ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
        """
        if fraud_probability < 0.25:
            return 'LOW'
        elif fraud_probability < 0.5:
            return 'MEDIUM'
        elif fraud_probability < 0.75:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        if self.model is None:
            return {'status': 'No model loaded'}
        
        return {
            'model_name': self.model_name,
            'model_stage': self.model_stage,
            'device': self.device,
            'model_summary': self.model.get_model_summary(),
            'total_predictions': len(self.prediction_history)
        }
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about recent predictions.
        
        Returns:
            Dict[str, Any]: Prediction statistics
        """
        if not self.prediction_history:
            return {'message': 'No predictions made yet'}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.prediction_history)
        
        stats = {
            'total_predictions': len(df),
            'fraud_predictions': df['prediction'].sum(),
            'fraud_rate': df['prediction'].mean(),
            'avg_fraud_probability': df['fraud_probability'].mean(),
            'median_fraud_probability': df['fraud_probability'].median(),
            'high_risk_transactions': (df['fraud_probability'] > 0.75).sum()
        }
        
        # Time-based statistics
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if len(df) > 1:
            time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600  # hours
            stats['predictions_per_hour'] = len(df) / max(time_span, 1/3600)  # Avoid division by zero
        
        return stats
    
    def clear_history(self) -> None:
        """Clear prediction history."""
        self.prediction_history.clear()
        logger.info("Prediction history cleared")
    
    def export_predictions(self, file_path: Path) -> None:
        """
        Export prediction history to CSV file.
        
        Args:
            file_path: Path to save the predictions
        """
        if not self.prediction_history:
            logger.warning("No predictions to export")
            return
        
        df = pd.DataFrame(self.prediction_history)
        df.to_csv(file_path, index=False)
        logger.info(f"Exported {len(df)} predictions to {file_path}")


def load_production_model() -> FraudPredictor:
    """
    Convenience function to load the production fraud detection model.
    
    Returns:
        FraudPredictor: Configured predictor with loaded model
    """
    predictor = FraudPredictor(
        model_name=config.MLFLOW_MODEL_NAME,
        model_stage="Production",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    try:
        predictor.load_model()
        logger.info("Production model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load production model: {str(e)}")
        logger.info("Falling back to latest available model")
        # Try to load any available model version
        client = MlflowClient()
        try:
            versions = client.get_latest_versions(config.MLFLOW_MODEL_NAME, stages=["None"])
            if versions:
                model_uri = f"models:/{config.MLFLOW_MODEL_NAME}/{versions[0].version}"
                predictor.load_model(model_uri=model_uri)
                logger.info(f"Loaded model version {versions[0].version}")
            else:
                raise ValueError("No model versions found")
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {str(fallback_error)}")
            raise
    
    return predictor


# Utility functions for transaction processing

def validate_transaction_input(transaction_data: Dict[str, Any]) -> bool:
    """
    Validate transaction input data.
    
    Args:
        transaction_data: Transaction data to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = ['sender_id', 'receiver_id', 'amount', 'type']
    
    for field in required_fields:
        if field not in transaction_data:
            logger.error(f"Missing required field: {field}")
            return False
    
    # Validate amount
    amount = transaction_data.get('amount', 0)
    if not isinstance(amount, (int, float)) or amount <= 0:
        logger.error(f"Invalid amount: {amount}")
        return False
    
    # Validate transaction type
    transaction_type = transaction_data.get('type', '')
    if transaction_type not in config.TRANSACTION_TYPES:
        logger.error(f"Invalid transaction type: {transaction_type}")
        return False
    
    return True


def create_transaction_summary(prediction_result: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of the prediction result.
    
    Args:
        prediction_result: Result from predict_fraud()
        
    Returns:
        str: Formatted summary string
    """
    fraud_prob = prediction_result.get('fraud_probability', 0)
    risk_level = prediction_result.get('risk_level', 'UNKNOWN')
    confidence = prediction_result.get('confidence', 0)
    
    summary = f"""
Transaction Analysis:
==================
Transaction ID: {prediction_result.get('transaction_id', 'N/A')}
Amount: ${prediction_result.get('amount', 0):,.2f}
Type: {prediction_result.get('type', 'N/A')}
Sender: {prediction_result.get('sender_id', 'N/A')}
Receiver: {prediction_result.get('receiver_id', 'N/A')}

Fraud Assessment:
================
Fraud Probability: {fraud_prob:.1%}
Risk Level: {risk_level}
Confidence: {confidence:.1%}
Prediction: {'🚨 FRAUD DETECTED' if fraud_prob > 0.5 else '✅ LEGITIMATE'}
"""
    
    return summary

