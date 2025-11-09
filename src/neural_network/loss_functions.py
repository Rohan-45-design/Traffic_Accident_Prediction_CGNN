"""
CGNN Loss Functions
===================
Combined loss with prediction and causal consistency
Updated with class weights to handle class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CGNNLoss(nn.Module):
    """
    Combined loss function for CGNN
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Loss weights
        self.pred_weight = config['training']['prediction_loss_weight']
        self.causal_weight = config['training']['causal_consistency_loss_weight']
        
        # Standard cross-entropy with label smoothing (NO class weights)
        self.prediction_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        print("✅ Using CrossEntropyLoss with label_smoothing=0.1 (NO class weights)")
        
    def forward(self, logits, targets, edge_index=None, edge_attr=None):
        """Compute combined loss"""
        pred_loss = self.prediction_loss(logits, targets)
        
        if edge_index is not None and self.causal_weight > 0:
            causal_loss = self.causal_consistency_loss(logits, edge_index, edge_attr)
        else:
            causal_loss = torch.tensor(0.0, device=logits.device)
        
        total_loss = (self.pred_weight * pred_loss + 
                     self.causal_weight * causal_loss)
        
        loss_dict = {
            'total': total_loss.item(),
            'prediction': pred_loss.item(),
            'causal': causal_loss.item()
        }
        
        return total_loss, loss_dict
    
    def causal_consistency_loss(self, logits, edge_index, edge_attr):
        """Causal consistency loss"""
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=logits.device)
        
        probs = torch.nn.functional.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        avg_entropy = torch.mean(entropy)
        
        if edge_attr is not None:
            avg_edge_strength = torch.mean(edge_attr)
            consistency_loss = avg_entropy * (1.0 / (avg_edge_strength + 1e-10))
        else:
            consistency_loss = avg_entropy
        
        return consistency_loss



class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Alternative to standard cross-entropy with class weights
    """
    
    def __init__(self, alpha=None, gamma=2.0):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Class weights [num_classes] or None
            gamma: Focusing parameter (default: 2.0)
        """
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                self.alpha = torch.tensor(self.alpha)
    
    def forward(self, logits, targets):
        """
        Compute focal loss
        
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: True labels [batch_size]
        
        Returns:
            focal_loss: Computed loss
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

class WeightedCGNNLoss(nn.Module):
    """
    Alternative CGNN Loss using Focal Loss instead of Cross Entropy
    Use this if standard class weights don't work well
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Loss weights
        self.pred_weight = config['training']['prediction_loss_weight']
        self.causal_weight = config['training']['causal_consistency_loss_weight']
        
        # Focal loss with class weights
        class_weights = [1.0, 1.0, 1.2]  # Extra weight for Class 2
        self.prediction_loss = FocalLoss(alpha=class_weights, gamma=2.0)
        print(f"✅ Using Focal Loss with alpha={class_weights}, gamma=2.0")
    
    def forward(self, logits, targets, edge_index=None, edge_attr=None):
        """
        Compute combined loss with focal loss
        
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: True labels [batch_size]
            edge_index: Graph edges [2, num_edges]
            edge_attr: Edge attributes [num_edges, 1]
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # Focal loss for predictions
        pred_loss = self.prediction_loss(logits, targets)
        
        # Causal consistency loss
        if edge_index is not None and self.causal_weight > 0:
            causal_loss = self._causal_consistency_loss(logits, edge_index, edge_attr)
        else:
            causal_loss = torch.tensor(0.0, device=logits.device)
        
        # Total loss
        total_loss = (self.pred_weight * pred_loss + 
                     self.causal_weight * causal_loss)
        
        # Return loss components
        loss_dict = {
            'total': total_loss.item(),
            'prediction': pred_loss.item(),
            'causal': causal_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _causal_consistency_loss(self, logits, edge_index, edge_attr):
        """Causal consistency loss"""
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=logits.device)
        
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        avg_entropy = torch.mean(entropy)
        
        if edge_attr is not None:
            avg_edge_strength = torch.mean(edge_attr)
            consistency_loss = avg_entropy * (1.0 / (avg_edge_strength + 1e-10))
        else:
            consistency_loss = avg_entropy
        
        return consistency_loss
