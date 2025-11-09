"""
Utility Functions
=================
Helper functions for CGNN implementation
"""

import torch
import numpy as np
import random
import yaml
import json
from pathlib import Path

def set_seed(seed=42):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to {seed}")

def count_parameters(model):
    """
    Count trainable parameters in model
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model has {total:,} trainable parameters")
    return total

def save_config(config, save_path):
    """
    Save configuration to file
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.endswith('.yaml') or save_path.endswith('.yml'):
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif save_path.endswith('.json'):
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"✅ Config saved to {save_path}")

def load_config(config_path):
    """
    Load configuration from file
    """
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")
    
    print(f"✅ Config loaded from {config_path}")
    return config

def get_device(config):
    """
    Get compute device (cuda or cpu)
    """
    device_name = config.get('device', 'cuda')
    
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"✅ Using device: {device}")
    
    return device

def create_directories(config):
    """
    Create necessary directories for outputs
    """
    dirs_to_create = [
        config['training']['checkpoint_dir'],
        config['output']['results_dir'],
        config['output']['logs_dir'],
        config['output']['plots_dir']
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created {len(dirs_to_create)} output directories")

def print_model_summary(model, input_data):
    """
    Print model architecture summary
    """
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*70)
    print(model)
    print("="*70)
    
    # Count parameters per layer
    print("\nParameters per layer:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name:50s} {param.numel():>10,}")
    
    print("="*70)
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print("="*70 + "\n")
