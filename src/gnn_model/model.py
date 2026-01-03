"""
Graph Neural Network Model for Fraud Detection.

This module implements a GraphSAGE-based model architecture for binary
fraud classification on transaction graphs. The model uses message passing
to aggregate neighborhood information and a multi-layer perceptron for
final classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, BatchNorm1d
from typing import Optional, Tuple, Dict, Any
import dgl
from dgl.nn import SAGEConv
import logging

from ..config import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class GraphSAGEClassifier(nn.Module):
    """
    GraphSAGE-based classifier for fraud detection.
    
    This model implements a two-layer GraphSAGE architecture followed by
    a multi-layer perceptron for binary classification. The model is designed
    to work with transaction graphs where nodes represent users and edges
    represent transactions.
    
    Architecture:
        1. Two GraphSAGE convolution layers with ReLU activation
        2. Dropout for regularization
        3. Multi-layer perceptron classifier
        4. Sigmoid activation for probability output
    
    Args:
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden layers in GraphSAGE
        output_dim: Dimension of GraphSAGE output
        num_layers: Number of GraphSAGE layers (default: 2)
        dropout_rate: Dropout probability (default: 0.2)
        classifier_hidden_dim: Hidden dimension of MLP classifier
        aggregator_type: GraphSAGE aggregator type ('mean', 'max', 'lstm')
    """
    
    def __init__(
        self,
        input_dim: int = config.GNN_INPUT_DIM,
        hidden_dim: int = config.GNN_HIDDEN_DIM,
        output_dim: int = config.GNN_OUTPUT_DIM,
        num_layers: int = config.GNN_NUM_LAYERS,
        dropout_rate: float = config.GNN_DROPOUT_RATE,
        classifier_hidden_dim: int = config.CLASSIFIER_HIDDEN_DIM,
        aggregator_type: str = 'mean'
    ) -> None:
        super(GraphSAGEClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.classifier_hidden_dim = classifier_hidden_dim
        self.aggregator_type = aggregator_type
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.convs.append(SAGEConv(
            input_dim, 
            hidden_dim, 
            aggregator_type=aggregator_type,
            feat_drop=dropout_rate
        ))
        self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Additional layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(
                hidden_dim,
                hidden_dim,
                aggregator_type=aggregator_type,
                feat_drop=dropout_rate
            ))
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Final GraphSAGE layer: hidden_dim -> output_dim
        if num_layers > 1:
            self.convs.append(SAGEConv(
                hidden_dim,
                output_dim,
                aggregator_type=aggregator_type,
                feat_drop=dropout_rate
            ))
            self.batch_norms.append(BatchNorm1d(output_dim))
        
        # Dropout layer
        self.dropout = Dropout(dropout_rate)
        
        # Multi-layer perceptron classifier
        classifier_input_dim = output_dim if num_layers > 1 else hidden_dim
        
        self.classifier = nn.Sequential(
            Linear(classifier_input_dim, classifier_hidden_dim),
            BatchNorm1d(classifier_hidden_dim),
            nn.ReLU(inplace=True),
            Dropout(dropout_rate),
            Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            BatchNorm1d(classifier_hidden_dim // 2),
            nn.ReLU(inplace=True),
            Dropout(dropout_rate),
            Linear(classifier_hidden_dim // 2, config.CLASSIFIER_OUTPUT_DIM)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"GraphSAGEClassifier initialized with {self.count_parameters()} parameters")
        logger.info(f"Architecture: {input_dim} -> {hidden_dim} -> {output_dim} -> {config.CLASSIFIER_OUTPUT_DIM}")
    
    def _init_weights(self) -> None:
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        g: dgl.DGLGraph, 
        node_features: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the GraphSAGE classifier.
        
        Args:
            g: DGL graph representing the transaction network
            node_features: Node feature tensor of shape [num_nodes, input_dim]
            edge_weight: Optional edge weights for weighted message passing
            
        Returns:
            torch.Tensor: Raw logits for binary classification [num_nodes, 1]
        """
        h = node_features
        
        # GraphSAGE message passing
        for i, conv in enumerate(self.convs):
            # Apply GraphSAGE convolution
            if edge_weight is not None:
                h = conv(g, h, edge_weight=edge_weight)
            else:
                h = conv(g, h)
            
            # Apply batch normalization
            if i < len(self.batch_norms):
                h = self.batch_norms[i](h)
            
            # Apply activation (except for the last layer)
            if i < len(self.convs) - 1:
                h = F.relu(h, inplace=True)
                h = self.dropout(h)
        
        # Final dropout before classifier
        h = self.dropout(h)
        
        # Apply MLP classifier
        logits = self.classifier(h)
        
        return logits
    
    def forward_with_attention(
        self,
        g: dgl.DGLGraph,
        node_features: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with attention weights for explainability.
        
        This method returns both predictions and intermediate representations
        that can be used for model interpretation and explainability.
        
        Args:
            g: DGL graph representing the transaction network
            node_features: Node feature tensor
            edge_weight: Optional edge weights
            
        Returns:
            Tuple containing:
                - logits: Raw classification logits
                - attention_dict: Dictionary with intermediate representations
        """
        h = node_features
        layer_outputs = []
        
        # Store input features
        attention_dict = {'input_features': h.detach().clone()}
        
        # GraphSAGE layers with intermediate storage
        for i, conv in enumerate(self.convs):
            if edge_weight is not None:
                h = conv(g, h, edge_weight=edge_weight)
            else:
                h = conv(g, h)
            
            # Store layer output before activation
            layer_outputs.append(h.detach().clone())
            
            if i < len(self.batch_norms):
                h = self.batch_norms[i](h)
            
            if i < len(self.convs) - 1:
                h = F.relu(h, inplace=True)
                h = self.dropout(h)
        
        # Store GraphSAGE outputs
        attention_dict['layer_outputs'] = layer_outputs
        attention_dict['graph_embeddings'] = h.detach().clone()
        
        h = self.dropout(h)
        logits = self.classifier(h)
        
        # Store classifier input
        attention_dict['classifier_input'] = h.detach().clone()
        
        return logits, attention_dict
    
    def get_node_embeddings(
        self,
        g: dgl.DGLGraph,
        node_features: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get node embeddings without classification head.
        
        This method returns the node embeddings from the GraphSAGE layers
        without passing through the classifier. Useful for visualization
        and analysis of learned representations.
        
        Args:
            g: DGL graph
            node_features: Node features
            edge_weight: Optional edge weights
            
        Returns:
            torch.Tensor: Node embeddings of shape [num_nodes, output_dim]
        """
        h = node_features
        
        for i, conv in enumerate(self.convs):
            if edge_weight is not None:
                h = conv(g, h, edge_weight=edge_weight)
            else:
                h = conv(g, h)
            
            if i < len(self.batch_norms):
                h = self.batch_norms[i](h)
            
            if i < len(self.convs) - 1:
                h = F.relu(h, inplace=True)
                h = self.dropout(h)
        
        return h
    
    def predict_proba(
        self,
        g: dgl.DGLGraph,
        node_features: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get fraud probabilities for nodes.
        
        Args:
            g: DGL graph
            node_features: Node features
            edge_weight: Optional edge weights
            
        Returns:
            torch.Tensor: Fraud probabilities [num_nodes, 1]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(g, node_features, edge_weight)
            probabilities = torch.sigmoid(logits)
        return probabilities
    
    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model architecture and parameters.
        
        Returns:
            Dict[str, Any]: Model summary information
        """
        total_params = self.count_parameters()
        
        # Count parameters by component
        gnn_params = sum(p.numel() for conv in self.convs for p in conv.parameters() if p.requires_grad)
        bn_params = sum(p.numel() for bn in self.batch_norms for p in bn.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        
        summary = {
            'model_name': 'GraphSAGEClassifier',
            'total_parameters': total_params,
            'gnn_parameters': gnn_params,
            'batch_norm_parameters': bn_params,
            'classifier_parameters': classifier_params,
            'architecture': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_layers': self.num_layers,
                'dropout_rate': self.dropout_rate,
                'classifier_hidden_dim': self.classifier_hidden_dim,
                'aggregator_type': self.aggregator_type
            }
        }
        
        return summary


class GraphDataLoader:
    """
    Data loader for graph-based fraud detection.
    
    This class handles the preparation of DGL graphs from pandas DataFrames
    and provides utilities for batching and sampling subgraphs.
    """
    
    def __init__(
        self,
        nodes_df: Optional[torch.Tensor] = None,
        edges_df: Optional[torch.Tensor] = None,
        device: str = 'cpu'
    ) -> None:
        """
        Initialize the GraphDataLoader.
        
        Args:
            nodes_df: DataFrame containing node information
            edges_df: DataFrame containing edge information  
            device: Device to place tensors on ('cpu' or 'cuda')
        """
        self.nodes_df = nodes_df
        self.edges_df = edges_df
        self.device = device
        self.graph: Optional[dgl.DGLGraph] = None
        
    def create_dgl_graph(
        self,
        nodes_df: torch.Tensor,
        edges_df: torch.Tensor,
        node_features: list[str],
        edge_features: list[str],
        target_column: str = 'is_fraud'
    ) -> dgl.DGLGraph:
        """
        Create a DGL graph from node and edge DataFrames.
        
        Args:
            nodes_df: Node DataFrame
            edges_df: Edge DataFrame
            node_features: List of column names to use as node features
            edge_features: List of column names to use as edge features
            target_column: Name of the target column for labels
            
        Returns:
            dgl.DGLGraph: Constructed DGL graph
        """
        import pandas as pd
        
        # Create user ID to index mapping
        unique_users = nodes_df['user_id'].unique()
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        
        # Map user IDs to indices in edges
        src_nodes = edges_df['source_user'].map(user_to_idx).values
        dst_nodes = edges_df['target_user'].map(user_to_idx).values
        
        # Create DGL graph
        g = dgl.graph((src_nodes, dst_nodes), num_nodes=len(unique_users))
        
        # Add node features
        node_feature_tensor = torch.tensor(
            nodes_df[node_features].values, 
            dtype=torch.float32,
            device=self.device
        )
        g.ndata['feat'] = node_feature_tensor
        
        # Add node labels (fraud rate as continuous target)
        if 'fraud_rate' in nodes_df.columns:
            node_labels = torch.tensor(
                nodes_df['fraud_rate'].values,
                dtype=torch.float32,
                device=self.device
            )
            g.ndata['label'] = node_labels
        
        # Add edge features
        edge_feature_tensor = torch.tensor(
            edges_df[edge_features].values,
            dtype=torch.float32,
            device=self.device
        )
        g.edata['feat'] = edge_feature_tensor
        
        # Add edge labels
        if target_column in edges_df.columns:
            edge_labels = torch.tensor(
                edges_df[target_column].values,
                dtype=torch.float32,
                device=self.device
            )
            g.edata['label'] = edge_labels
        
        # Add edge weights
        if 'weight' in edges_df.columns:
            edge_weights = torch.tensor(
                edges_df['weight'].values,
                dtype=torch.float32,
                device=self.device
            )
            g.edata['weight'] = edge_weights
        
        self.graph = g
        logger.info(f"Created DGL graph with {g.num_nodes()} nodes and {g.num_edges()} edges")
        
        return g
    
    def get_subgraph(
        self,
        node_ids: list[int],
        num_hops: int = 2
    ) -> dgl.DGLGraph:
        """
        Extract a subgraph around specified nodes.
        
        Args:
            node_ids: List of node IDs to center the subgraph around
            num_hops: Number of hops for subgraph extraction
            
        Returns:
            dgl.DGLGraph: Extracted subgraph
        """
        if self.graph is None:
            raise ValueError("Graph not created. Call create_dgl_graph() first.")
        
        # Sample subgraph
        subgraph_nodes = dgl.khop_subgraph(self.graph, node_ids, num_hops)[0]
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        return subgraph
    
    def split_graph(
        self,
        train_ratio: float = config.TRAIN_RATIO,
        val_ratio: float = config.VAL_RATIO,
        test_ratio: float = config.TEST_RATIO,
        random_state: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split nodes into train, validation, and test sets.
        
        Args:
            train_ratio: Proportion of nodes for training
            val_ratio: Proportion of nodes for validation  
            test_ratio: Proportion of nodes for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of masks for train, validation, and test nodes
        """
        if self.graph is None:
            raise ValueError("Graph not created. Call create_dgl_graph() first.")
        
        num_nodes = self.graph.num_nodes()
        
        # Generate random permutation
        torch.manual_seed(random_state)
        perm = torch.randperm(num_nodes)
        
        # Calculate split indices
        train_end = int(train_ratio * num_nodes)
        val_end = train_end + int(val_ratio * num_nodes)
        
        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[perm[:train_end]] = True
        val_mask[perm[train_end:val_end]] = True
        test_mask[perm[val_end:]] = True
        
        return train_mask, val_mask, test_mask

