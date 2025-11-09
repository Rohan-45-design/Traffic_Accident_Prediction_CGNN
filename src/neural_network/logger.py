"""
Training Logger
===============
Log training progress and metrics
"""

import os
from pathlib import Path
from datetime import datetime

class TrainingLogger:
    """
    Log training metrics and progress
    """
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'training_log_{timestamp}.txt'
        
        self.epoch_metrics = []
        
        # Write header
        self.log(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("="*70)
    
    def log(self, message):
        """Log message to file and console"""
        print(message)
        with open(self.log_file, 'a',encoding='utf=8') as f:
            f.write(message + '\n')
    
    def log_epoch(self, epoch, metrics):
        """
        Log metrics for an epoch
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        self.epoch_metrics.append({'epoch': epoch, **metrics})
        
        # Format message
        msg = f"Epoch {epoch:3d} | "
        msg += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        self.log(msg)
    
    def log_final_results(self, results):
        """Log final test results"""
        self.log("\n" + "="*70)
        self.log("FINAL RESULTS")
        self.log("="*70)
        
        for key, value in results.items():
            self.log(f"{key:20s}: {value:.4f}")
        
        self.log("="*70)
    
    def save_metrics(self):
        """Save all metrics to CSV"""
        import pandas as pd
        
        if self.epoch_metrics:
            df = pd.DataFrame(self.epoch_metrics)
            csv_path = self.log_dir / 'training_metrics.csv'
            df.to_csv(csv_path, index=False)
            self.log(f"\n✅ Metrics saved to {csv_path}")
