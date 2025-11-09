"""
Core CGNN Model - FIXED
=======================
Complete Causal Graph Neural Network architecture
Processes samples with feature-level causal graph
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

class FeatureGraphLayer(nn.Module):
    """
    Process features through causal graph for a single sample
    """
    def __init__(self, hidden_dim, edge_index, edge_attr):
        super().__init__()
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.hidden_dim = hidden_dim
        
        # Simple message passing
        self.message_nn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        """
        x: [num_features, hidden_dim]
        Returns: [num_features, hidden_dim]
        """
        # For each edge, pass messages
        messages = []
        for i in range(self.edge_index.size(1)):
            src = self.edge_index[0, i]
            dst = self.edge_index[1, i]
            edge_w = self.edge_attr[i, 0]
            
            # Concatenate source and target features
            msg_input = torch.cat([x[src], x[dst]], dim=0)
            msg = self.message_nn(msg_input) * edge_w
            messages.append((dst.item(), msg))
        
        # Aggregate messages
        out = x.clone()
        for dst, msg in messages:
            out[dst] = out[dst] + msg
        
        return out

class CGNN(nn.Module):
    """
    Simplified CGNN for per-sample processing
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.input_dim = config['model']['input_dim']
        self.hidden_dim = config['model']['hidden_dim']
        self.output_dim = config['model']['output_dim']
        self.num_layers = config['model']['num_layers']
        self.dropout = config['model']['dropout']
        
        # Feature projection (per feature)
        self.feature_encoder = nn.Linear(1, self.hidden_dim)
        
        # Graph processing layers  
        self.graph_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            for _ in range(self.num_layers)
        ])
        
        # Global pooling and classification
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # Store graph structure (will be set during first forward pass)
        self.edge_index = None
        self.edge_attr = None
        
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [batch_size, num_features]
                - edge_index: Graph edges [2, num_edges]
                - edge_attr: Edge attributes [num_edges, 1]
        
        Returns:
            logits: Predictions [batch_size, output_dim]
            attention_weights: Empty list (for compatibility)
        """
        x = data.x  # [batch_size, num_features]
        
        # Store graph structure
        if self.edge_index is None:
            self.edge_index = data.edge_index.to(x.device)
            self.edge_attr = data.edge_attr.to(x.device)
        
        batch_size = x.size(0)
        num_features = x.size(1)
        
        # Process each sample
        outputs = []
        for i in range(batch_size):
            sample = x[i]  # [num_features]
            
            # Encode each feature
            h = self.feature_encoder(sample.unsqueeze(1))  # [num_features, hidden_dim]
            
            # Apply graph layers with message passing
            for layer in self.graph_layers:
                h_new = layer(h)
                
                # Simple message passing using edge_index
                if self.edge_index.size(1) > 0:
                    messages = torch.zeros_like(h_new)
                    for edge_idx in range(self.edge_index.size(1)):
                        src = self.edge_index[0, edge_idx]
                        dst = self.edge_index[1, edge_idx]
                        weight = self.edge_attr[edge_idx, 0]
                        
                        if src < num_features and dst < num_features:
                            messages[dst] += h_new[src] * weight
                    
                    h = h + messages
                else:
                    h = h_new
                
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            
            # Flatten and classify
            h_flat = h.view(-1)  # [num_features * hidden_dim]
            outputs.append(h_flat)
        
        # Stack all samples
        h_batch = torch.stack(outputs)  # [batch_size, num_features * hidden_dim]
        
        # Classification
        logits = self.classifier(h_batch)
        
        return logits, []
    
    def predict(self, data):
        """Make predictions (returns class labels)"""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(data)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def predict_proba(self, data):
        """Get prediction probabilities"""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(data)
            probs = F.softmax(logits, dim=1)
        return probs
