"""
Causal Message Passing Layer
=============================
Message passing that respects causal relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class CausalMessagePassing(MessagePassing):
    """
    Causal-aware message passing layer
    Aggregates information from causally related nodes
    """
    
    def __init__(self, in_channels, out_channels, aggr='mean', 
                 use_edge_attr=True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_edge_attr = use_edge_attr
        
        # Message function
        if use_edge_attr:
            self.message_nn = nn.Sequential(
                nn.Linear(in_channels * 2 + 1, out_channels),  # [src, dst, edge_attr]
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
        else:
            self.message_nn = nn.Sequential(
                nn.Linear(in_channels * 2, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
        
        # Update function
        self.update_nn = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.message_nn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        for layer in self.update_nn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, 1]
        
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Add self-loops for stability
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value=1.0, num_nodes=x.size(0)
        )
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        return out
    
    def message(self, x_i, x_j, edge_attr):
        """
        Construct messages from source nodes to target nodes
        
        Args:
            x_i: Target node features [num_edges, in_channels]
            x_j: Source node features [num_edges, in_channels]
            edge_attr: Edge attributes [num_edges, 1]
        
        Returns:
            Messages [num_edges, out_channels]
        """
        if self.use_edge_attr and edge_attr is not None:
            # Concatenate [source, target, edge_weight]
            msg_input = torch.cat([x_j, x_i, edge_attr], dim=-1)
        else:
            # Concatenate [source, target]
            msg_input = torch.cat([x_j, x_i], dim=-1)
        
        # Transform through message network
        message = self.message_nn(msg_input)
        
        return message
    
    def update(self, aggr_out, x):
        """
        Update node features with aggregated messages
        
        Args:
            aggr_out: Aggregated messages [num_nodes, out_channels]
            x: Original node features [num_nodes, in_channels]
        
        Returns:
            Updated features [num_nodes, out_channels]
        """
        # Concatenate original features with aggregated messages
        update_input = torch.cat([x, aggr_out], dim=-1)
        
        # Transform through update network
        out = self.update_nn(update_input)
        
        return out

class ResidualCausalLayer(nn.Module):
    """
    Causal message passing with residual connection
    """
    
    def __init__(self, channels, aggr='mean', dropout=0.3):
        super().__init__()
        
        self.message_passing = CausalMessagePassing(
            channels, channels, aggr=aggr
        )
        self.batch_norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward with residual connection
        """
        # Message passing
        out = self.message_passing(x, edge_index, edge_attr)
        
        # Residual connection
        out = out + x
        
        # Normalize and dropout
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        return out
