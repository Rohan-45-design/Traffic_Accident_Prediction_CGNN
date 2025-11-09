"""
CGNN Evaluator
==============
Evaluate trained CGNN model
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class CGNNEvaluator:
    """
    Evaluate CGNN model performance
    """
    
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.num_classes = config['model']['output_dim']
    
    def evaluate(self, test_data):
        """
        Evaluate model on test data
        
        Args:
            test_data: PyTorch Geometric Data object
        
        Returns:
            results: Dictionary of evaluation metrics
        """
        self.model.eval()
        test_data = test_data.to(self.device)
        
        with torch.no_grad():
            # Get predictions
            logits, _ = self.model(test_data)
            predictions = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)
            
            # Move to CPU for sklearn
            y_true = test_data.y.cpu().numpy()
            y_pred = predictions.cpu().numpy()
            y_prob = probabilities.cpu().numpy()
        
        # Compute metrics
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Per-class metrics
        for i in range(self.num_classes):
            mask = y_true == i
            if mask.sum() > 0:
                class_acc = (y_pred[mask] == y_true[mask]).mean()
                results[f'class_{i}_accuracy'] = class_acc
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm
        
        # Classification report
        report = classification_report(y_true, y_pred, zero_division=0)
        results['classification_report'] = report
        
        return results, y_true, y_pred, y_prob
    
    def print_results(self, results):
        """Print evaluation results"""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1 Score:  {results['f1']:.4f}")
        print("\nPer-Class Accuracy:")
        for i in range(self.num_classes):
            if f'class_{i}_accuracy' in results:
                print(f"  Class {i}: {results[f'class_{i}_accuracy']:.4f}")
        print("\nClassification Report:")
        print(results['classification_report'])
        print("="*70 + "\n")
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=[f'Class {i}' for i in range(self.num_classes)],
            yticklabels=[f'Class {i}' for i in range(self.num_classes)],
            cbar_kws={'label': 'Percentage'}
        )
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_class_distribution(self, y_true, y_pred, save_path=None):
        """
        Plot true vs predicted class distribution
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # True distribution
        unique, counts = np.unique(y_true, return_counts=True)
        ax1.bar(unique, counts, color='steelblue', alpha=0.8)
        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('True Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(self.num_classes))
        
        # Predicted distribution
        unique, counts = np.unique(y_pred, return_counts=True)
        ax2.bar(unique, counts, color='coral', alpha=0.8)
        ax2.set_xlabel('Class', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Predicted Class Distribution', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(self.num_classes))
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Class distribution plot saved to {save_path}")
        
        plt.close()
    
    def save_results(self, results, save_path):
        """Save evaluation results to file"""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                results_serializable[key] = float(value)
            else:
                results_serializable[key] = value
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"✅ Results saved to {save_path}")
