"""
Main Script - Phase 4
======================
Execute CGNN training and evaluation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

import torch
from src.neural_network import (
    CGNNDataLoader,
    CGNN,
    CGNNTrainer,
    CGNNEvaluator,
    set_seed,
    count_parameters,
    get_device,
    create_directories,
    load_config,
    print_model_summary
)

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("CGNN - PHASE 4: MODEL TRAINING")
    print("="*70 + "\n")
    
    # Load configuration
    config = load_config('configs/phase4_config.yaml')
    
    # Set random seed
    set_seed(config['data']['random_seed'])
    
    # Get device
    device = get_device(config)
    
    # Create output directories
    create_directories(config)
    
    # ========================================
    # STEP 1: Load and prepare data
    # ========================================
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING")
    print("="*70)
    
    data_loader = CGNNDataLoader(config)
    train_data, val_data, test_data, feature_names = data_loader.prepare_data()
    
    # Save scaler
    scaler_path = Path(config['output']['results_dir']) / 'scaler.pkl'
    data_loader.save_scaler(scaler_path)
    
    # ========================================
    # STEP 2: Create model
    # ========================================
    print("\n" + "="*70)
    print("STEP 2: MODEL CREATION")
    print("="*70)
    
    model = CGNN(config)
    print_model_summary(model, train_data)
    
    # ========================================
    # STEP 3: Train model
    # ========================================
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70)
    
    trainer = CGNNTrainer(model, config, device)
    best_metrics = trainer.train(train_data, val_data)
    
    # Load best model
    trainer.load_best_model()
    
    # ========================================
    # STEP 4: Evaluate model
    # ========================================
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION")
    print("="*70)
    
    evaluator = CGNNEvaluator(model, config, device)
    results, y_true, y_pred, y_prob = evaluator.evaluate(test_data)
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    results_path = Path(config['output']['results_dir']) / 'test_results.json'
    evaluator.save_results(results, results_path)
    
    # Plot confusion matrix
    cm_path = Path(config['output']['plots_dir']) / 'confusion_matrix.png'
    evaluator.plot_confusion_matrix(results['confusion_matrix'], cm_path)
    
    # Plot class distribution
    dist_path = Path(config['output']['plots_dir']) / 'class_distribution.png'
    evaluator.plot_class_distribution(y_true, y_pred, dist_path)
    
    # ========================================
    # STEP 5: Save final model
    # ========================================
    print("\n" + "="*70)
    print("STEP 5: SAVING MODEL")
    print("="*70)
    
    model_save_path = config['output']['model_save_path']
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'feature_names': feature_names,
        'test_results': results
    }, model_save_path)
    print(f"✅ Model saved to {model_save_path}")
    
    # ========================================
    # DONE
    # ========================================
    print("\n" + "="*70)
    print("✅ PHASE 4 COMPLETE!")
    print("="*70)
    print(f"\nFinal Test Accuracy: {results['accuracy']:.4f}")
    print(f"Final Test F1 Score: {results['f1']:.4f}")
    print("\nNext: Run Phase 5 for interventional predictions and explainability")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
