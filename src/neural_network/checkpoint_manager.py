"""
Checkpoint Manager
==================
Save and load model checkpoints
"""

import torch
from pathlib import Path
import shutil

class CheckpointManager:
    """
    Manage model checkpoints during training
    """
    
    def __init__(self, checkpoint_dir, save_best_only=True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.best_metric = float('inf')
    
    def save_checkpoint(self, epoch, model, optimizer, metrics, is_best=False):
        """
        Save checkpoint
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save regular checkpoint
        if not self.save_best_only:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"✅ Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"⭐ Best model saved: {best_path}")
            self.best_metric = metrics.get('val_loss', float('inf'))
    
    def load_checkpoint(self, model, optimizer=None, checkpoint_name='best_model.pt'):
        """
        Load checkpoint
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            checkpoint_name: Name of checkpoint file
        
        Returns:
            epoch: Epoch of loaded checkpoint
            metrics: Metrics from checkpoint
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return 0, {}
        
        checkpoint = torch.load(checkpoint_path, weights_only=False)  # Fixed for PyTorch 2.6+
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        print(f"✅ Checkpoint loaded from: {checkpoint_path}")
        print(f"   Epoch: {epoch}")
        print(f"   Metrics: {metrics}")
        
        return epoch, metrics

    
    def get_best_checkpoint_path(self):
        """Get path to best model checkpoint"""
        return self.checkpoint_dir / 'best_model.pt'
