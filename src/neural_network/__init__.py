"""
Neural Network Package
======================
CGNN implementation for traffic accident prediction
"""

from .data_loader import CGNNDataLoader
from .causal_attention import CausalAttentionLayer
from .message_passing import CausalMessagePassing, ResidualCausalLayer
from .cgnn_model import CGNN
from .loss_functions import CGNNLoss, FocalLoss
from .trainer import CGNNTrainer
from .evaluator import CGNNEvaluator
from .checkpoint_manager import CheckpointManager
from .logger import TrainingLogger
from .utils import (
    set_seed, 
    count_parameters, 
    save_config, 
    load_config,
    get_device,
    create_directories,
    print_model_summary
)

__all__ = [
    # Data
    'CGNNDataLoader',
    
    # Model Components
    'CausalAttentionLayer',
    'CausalMessagePassing',
    'ResidualCausalLayer',
    'CGNN',
    
    # Training
    'CGNNLoss',
    'FocalLoss',
    'CGNNTrainer',
    'CGNNEvaluator',
    
    # Utilities
    'CheckpointManager',
    'TrainingLogger',
    'set_seed',
    'count_parameters',
    'save_config',
    'load_config',
    'get_device',
    'create_directories',
    'print_model_summary'
]

__version__ = '1.0.0'
