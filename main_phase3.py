
import pandas as pd
import numpy as np
import json
import yaml
import sys
import warnings
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from causal_discovery.pc_algorithm import PCAlgorithm
from causal_discovery.causal_validator import CausalValidator  
from causal_discovery.graph_constructor import GraphConstructor
from causal_discovery.causal_visualizer import CausalVisualizer

warnings.filterwarnings('ignore')

def load_config():
    """Load Phase 3 configuration"""
    config_path = Path("configs/phase3_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration
        return {
            'pc_algorithm': {
                'alpha': 0.05,
                'max_cond_vars': 3,
                'method': 'fisherz'
            },
            'validation': {
                'bootstrap_samples': 100,
                'confidence_level': 0.95
            },
            'output': {
                'save_graphs': True,
                'create_visualizations': True
            }
        }

def main_phase3_pipeline():
    """Execute complete Phase 3: Causal Discovery"""
    
    print("🚀 CGNN PROJECT - PHASE 3: CAUSAL DISCOVERY IMPLEMENTATION")
    print("="*70)
    
    # Load configuration
    config = load_config()
    
    try:
        # STEP 1: Load Causal Variables
        print("\n📊 STEP 1: LOADING CAUSAL VARIABLES")
        print("-" * 50)
        
        causal_data_path = Path("data/processed/causal_variables.csv")
        if not causal_data_path.exists():
            raise FileNotFoundError(f"❌ Causal variables file not found: {causal_data_path}")
        
        causal_df = pd.read_csv(causal_data_path)
        print(f"📥 Loaded causal variables: {causal_df.shape}")
        print(f"📋 Variables: {list(causal_df.columns)}")
        
        # Validate data
        if len(causal_df) < 100:
            print("⚠️ Warning: Small dataset may affect causal discovery reliability")
        
        # STEP 2: PC Algorithm Causal Discovery
        print("\n🧠 STEP 2: PC ALGORITHM CAUSAL DISCOVERY")
        print("-" * 50)
        
        pc_algorithm = PCAlgorithm(
            alpha=config['pc_algorithm']['alpha'],
            max_cond_vars=config['pc_algorithm']['max_cond_vars'],
            method=config['pc_algorithm']['method']
        )
        
        causal_graph, relationships = pc_algorithm.discover_causal_structure(causal_df)
        print(f"✅ Causal discovery complete: {len(relationships)} relationships found")
        
        # STEP 3: Causal Validation
        print("\n📊 STEP 3: CAUSAL RELATIONSHIP VALIDATION")
        print("-" * 50)
        
        validator = CausalValidator(
            bootstrap_samples=config['validation']['bootstrap_samples'],
            confidence_level=config['validation']['confidence_level']
        )
        
        validation_results = validator.validate_relationships(causal_df, relationships)
        print(f"✅ Validation complete: {validation_results['validated_count']} relationships validated")
        
        # STEP 4: Graph Construction
        print("\n🔗 STEP 4: CAUSAL GRAPH CONSTRUCTION")
        print("-" * 50)
        
        constructor = GraphConstructor()
        structured_graph = constructor.build_causal_graph(relationships, list(causal_df.columns))
        print(f"✅ Graph constructed: {len(structured_graph.nodes)} nodes, {len(structured_graph.edges)} edges")
        
        # STEP 5: Visualization & Results
        print("\n📈 STEP 5: VISUALIZATION & RESULTS")
        print("-" * 50)
        
        visualizer = CausalVisualizer()
        
        if config['output']['create_visualizations']:
            viz_path = visualizer.create_graph_visualization(structured_graph, "data/causal_graphs/causal_graph_viz.png")
            print(f"📊 Visualization saved: {viz_path}")
        
        # STEP 6: Save Results
        print("\n💾 STEP 6: SAVING RESULTS")
        print("-" * 50)
        
        # Create output directories
        Path("data/causal_graphs").mkdir(parents=True, exist_ok=True)
        Path("results").mkdir(exist_ok=True)
        
        # Save causal graph
        import pickle
        with open("data/causal_graphs/causal_graph.pkl", "wb") as f:
            pickle.dump(structured_graph, f)
        
        # Save relationships
        relationships_df = pd.DataFrame(relationships)
        relationships_df.to_csv("data/causal_graphs/causal_relationships.csv", index=False)
        
        # Save validation results
        with open("data/causal_graphs/validation_report.json", "w") as f:
            json.dump(validation_results, f, indent=2)
        
        # Create summary results
        summary_results = {
            'phase': 3,
            'status': 'completed',
            'input_variables': len(causal_df.columns),
            'sample_size': len(causal_df),
            'discovered_relationships': len(relationships),
            'validated_relationships': validation_results['validated_count'],
            'graph_nodes': len(structured_graph.nodes),
            'graph_edges': len(structured_graph.edges),
            'config': config
        }
        
        with open("results/phase3_results.json", "w") as f:
            json.dump(summary_results, f, indent=2)
        
        # COMPLETION SUMMARY
        print("\n📋 PHASE 3 COMPLETION SUMMARY")
        print("="*70)
        print(f"📊 Input variables: {len(causal_df.columns)}")
        print(f"📊 Sample size: {len(causal_df):,}")
        print(f"🧠 Discovered relationships: {len(relationships)}")
        print(f"✅ Validated relationships: {validation_results['validated_count']}")
        print(f"🔗 Graph nodes: {len(structured_graph.nodes)}")
        print(f"🔗 Graph edges: {len(structured_graph.edges)}")
        
        print("\n📁 Generated Files:")
        print("✅ data/causal_graphs/causal_graph.pkl")
        print("✅ data/causal_graphs/causal_relationships.csv")
        print("✅ data/causal_graphs/validation_report.json")
        print("✅ data/causal_graphs/causal_graph_viz.png")
        print("✅ results/phase3_results.json")
        
        print("\n🎉 PHASE 3 COMPLETED SUCCESSFULLY!")
        print("🚀 Ready to proceed to Phase 4: Neural Architecture Development")
        
        return summary_results
        
    except Exception as e:
        print(f"❌ Phase 3 execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main_phase3_pipeline()
