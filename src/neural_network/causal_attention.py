"""
Causal Attention Mechanism
===========================
Multi-head attention that respects causal structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class CausalAttentionLayer(MessagePassing):
    """
    Causal multi-head attention layer
    Applies attention based on causal graph structure
    """
    
    def __init__(self, in_channels, out_channels, num_heads=4, 
                 dropout=0.2, edge_dim=1, **kwargs):
        super().__init__(aggr='add', **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        self.head_dim = out_channels // num_heads
        
        # Q, K, V projections
        self.lin_query = nn.Linear(in_channels, out_channels)
        self.lin_key = nn.Linear(in_channels, out_channels)
        self.lin_value = nn.Linear(in_channels, out_channels)
        
        # Edge encoding (causal strength)
        self.edge_encoder = nn.Linear(edge_dim, num_heads)
        
        # Output projection
        self.lin_out = nn.Linear(out_channels, out_channels)
        
        self.attn_dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.lin_query.weight)
        nn.init.xavier_uniform_(self.lin_key.weight)
        nn.init.xavier_uniform_(self.lin_value.weight)
        nn.init.xavier_uniform_(self.edge_encoder.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)
        
        if self.lin_query.bias is not None:
            nn.init.zeros_(self.lin_query.bias)
            nn.init.zeros_(self.lin_key.bias)
            nn.init.zeros_(self.lin_value.bias)
            nn.init.zeros_(self.edge_encoder.bias)
            nn.init.zeros_(self.lin_out.bias)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
        
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Project to Q, K, V
        query = self.lin_query(x).view(-1, self.num_heads, self.head_dim)
        key = self.lin_key(x).view(-1, self.num_heads, self.head_dim)
        value = self.lin_value(x).view(-1, self.num_heads, self.head_dim)
        
        # Message passing with attention
        out = self.propagate(
            edge_index, 
            query=query, 
            key=key, 
            value=value, 
            edge_attr=edge_attr,
            size=None
        )
        
        # Output projection
        out = out.view(-1, self.out_channels)
        out = self.lin_out(out)
        
        return out
    
    def message(self, query_i, key_j, value_j, edge_attr, index, ptr, size_i):
        """
        Compute attention-weighted messages
        
        Args:
            query_i: Query vectors of target nodes [num_edges, num_heads, head_dim]
            key_j: Key vectors of source nodes [num_edges, num_heads, head_dim]
            value_j: Value vectors of source nodes [num_edges, num_heads, head_dim]
            edge_attr: Edge attributes (causal strength) [num_edges, edge_dim]
            index: Target node indices
            ptr: Aggregation pointer
            size_i: Number of target nodes
        
        Returns:
            Attention-weighted messages [num_edges, num_heads, head_dim]
        """
        # Compute attention scores: Q * K^T / sqrt(d)
        scores = (query_i * key_j).sum(dim=-1) / (self.head_dim ** 0.5)  # [num_edges, num_heads]
        
        # Encode causal strength into attention
        if edge_attr is not None:
            edge_scores = self.edge_encoder(edge_attr)  # [num_edges, num_heads]
            scores = scores * torch.sigmoid(edge_scores)  # Weight by causal strength
        
        # Apply softmax over neighbors
        alpha = softmax(scores, index, ptr, size_i)  # [num_edges, num_heads]
        alpha = self.attn_dropout(alpha)
        
        # Weighted sum of values
        out = value_j * alpha.unsqueeze(-1)  # [num_edges, num_heads, head_dim]
        
        return out
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        Aggregate messages
        
        Args:
            inputs: Messages [num_edges, num_heads, head_dim]
            index: Target node indices
            ptr: Aggregation pointer
            dim_size: Number of target nodes
        
        Returns:
            Aggregated features [num_nodes, num_heads, head_dim]
        """
        # Sum aggregation
        out = torch.zeros(
            (dim_size if dim_size is not None else index.max().item() + 1, 
             inputs.size(1), 
             inputs.size(2)),
            dtype=inputs.dtype,
            device=inputs.device
        )
        out.index_add_(0, index, inputs)
        
        return out
    
    def update(self, aggr_out):
        """
        Update step (identity function - just return aggregated output)
        
        Args:
            aggr_out: Aggregated messages [num_nodes, num_heads, head_dim]
        
        Returns:
            Updated features [num_nodes, num_heads, head_dim]
        """
        return aggr_out
