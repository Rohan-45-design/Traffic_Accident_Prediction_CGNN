"""
Test Model Script
=================
Quick test of model functionality
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from src.neural_network import (
    CGNNDataLoader,
    CGNN,
    load_config,
    set_seed,
    get_device
)

def test_model():
    """Test model can load and make predictions"""
    
    print("\n" + "="*70)
    print("TESTING CGNN MODEL")
    print("="*70 + "\n")
    
    # Load config
    config = load_config('configs/phase4_config.yaml')
    set_seed(42)
    device = get_device(config)
    
    # Load data
    print("Loading data...")
    data_loader = CGNNDataLoader(config)
    train_data, val_data, test_data, feature_names = data_loader.prepare_data()
    
    # Create model
    print("\nCreating model...")
    model = CGNN(config).to(device)
    
    # Test forward pass
    print("\nTesting forward pass...")
    test_data = test_data.to(device)
    
    model.eval()
    with torch.no_grad():
        logits, attention_weights = model(test_data)
        predictions = torch.argmax(logits, dim=1)
    
    print(f"✅ Forward pass successful!")
    print(f"   Input shape: {test_data.x.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Unique predictions: {predictions.unique().tolist()}")
    
    # Test prediction methods
    print("\nTesting prediction methods...")
    preds = model.predict(test_data)
    probs = model.predict_proba(test_data)
    
    print(f"✅ Predictions: {preds.shape}")
    print(f"✅ Probabilities: {probs.shape}")
    print(f"   Probability sum: {probs.sum(dim=1)[0]:.4f} (should be ~1.0)")
    
    # Test on single sample
    print("\nTesting single sample prediction...")
    single_sample = test_data.clone()
    single_sample.x = test_data.x[:1]  # Take first sample
    
    pred_single = model.predict(single_sample)
    prob_single = model.predict_proba(single_sample)
    
    print(f"✅ Single prediction: Class {pred_single.item()}")
    print(f"✅ Class probabilities: {prob_single[0].tolist()}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_model()
