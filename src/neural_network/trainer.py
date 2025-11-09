"""
CGNN Trainer
============
Training loop for CGNN model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from .logger import TrainingLogger
from .checkpoint_manager import CheckpointManager

class CGNNTrainer:
    """
    Trainer class for CGNN
    """
    
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss function
        from .loss_functions import CGNNLoss
        self.criterion = CGNNLoss(config)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            config['training']['checkpoint_dir'],
            save_best_only=True
        )
        
        # Logger
        self.logger = TrainingLogger(config['output']['logs_dir'])
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def _create_optimizer(self):
        """Create optimizer"""
        optimizer_name = self.config['training']['optimizer']
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_name = self.config['training'].get('scheduler', 'ReduceLROnPlateau')
        
        if scheduler_name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config['training']['scheduler_factor'],
                patience=self.config['training']['scheduler_patience'],
                #verbose=True
            )
        elif scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, train_data):
        """
        Train for one epoch
        
        Args:
            train_data: PyTorch Geometric Data object
        
        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        
        # Move data to device
        train_data = train_data.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        logits, _ = self.model(train_data)
        
        # Compute loss
        loss, loss_dict = self.criterion(
            logits, 
            train_data.y,
            train_data.edge_index,
            train_data.edge_attr
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        # Compute accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == train_data.y).float().mean().item()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy,
            **loss_dict
        }
        
        return metrics
    
    def validate(self, val_data):
        """
        Validate model
        
        Args:
            val_data: PyTorch Geometric Data object
        
        Returns:
            metrics: Dictionary of validation metrics
        """
        self.model.eval()
        
        val_data = val_data.to(self.device)
        
        with torch.no_grad():
            # Forward pass
            logits, _ = self.model(val_data)
            
            # Compute loss
            loss, loss_dict = self.criterion(
                logits,
                val_data.y,
                val_data.edge_index,
                val_data.edge_attr
            )
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == val_data.y).float().mean().item()
            
            # Compute per-class accuracy
            num_classes = logits.size(1)
            class_acc = []
            for c in range(num_classes):
                mask = val_data.y == c
                if mask.sum() > 0:
                    class_acc.append(
                        (predictions[mask] == val_data.y[mask]).float().mean().item()
                    )
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy,
            'class_accuracy': np.mean(class_acc) if class_acc else 0.0,
            **loss_dict
        }
        
        return metrics
    
    def train(self, train_data, val_data, num_epochs=None):
        """
        Full training loop
        
        Args:
            train_data: Training data
            val_data: Validation data
            num_epochs: Number of epochs (default from config)
        
        Returns:
            best_metrics: Best validation metrics
        """
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        self.logger.log("\n" + "="*70)
        self.logger.log("STARTING TRAINING")
        self.logger.log("="*70)
        self.logger.log(f"Epochs: {num_epochs}")
        self.logger.log(f"Device: {self.device}")
        self.logger.log(f"Optimizer: {self.config['training']['optimizer']}")
        self.logger.log(f"Learning Rate: {self.config['training']['learning_rate']}")
        self.logger.log("="*70 + "\n")
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_data)
            
            # Validate
            val_metrics = self.validate(val_data)
            
            # Log metrics
            if epoch % self.config['training']['log_interval'] == 0:
                metrics_to_log = {
                    'train_loss': train_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_acc': val_metrics['accuracy']
                }
                self.logger.log_epoch(epoch, metrics_to_log)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Check for improvement
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if epoch % self.config['training']['save_interval'] == 0 or is_best:
                self.checkpoint_manager.save_checkpoint(
                    epoch,
                    self.model,
                    self.optimizer,
                    val_metrics,
                    is_best=is_best
                )
            
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                self.logger.log(f"\n⚠️ Early stopping triggered at epoch {epoch}")
                break
        
        self.logger.log("\n" + "="*70)
        self.logger.log("TRAINING COMPLETE")
        self.logger.log("="*70)
        self.logger.log(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save metrics
        self.logger.save_metrics()
        
        return {'best_val_loss': self.best_val_loss}
    
    def load_best_model(self):
        """Load the best model from checkpoint"""
        self.checkpoint_manager.load_checkpoint(self.model, self.optimizer)
