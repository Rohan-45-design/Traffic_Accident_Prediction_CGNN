"""
CGNN Data Loader
================
Load and prepare data for CGNN training
"""

import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle

class CGNNDataLoader:
    """Load and prepare data for CGNN"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = None
        self.num_classes = config['model']['output_dim']
        
    def load_data(self):
        """Load causal variables and relationships"""
        print("="*70)
        print("📊 LOADING DATA")
        print("="*70)
        
        # Load features
        data_path = self.config['data']['causal_variables']
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} samples with {len(df.columns)} features")
        print(f"   Features: {list(df.columns)}")
        
        # Load causal graph
        edges_path = self.config['data']['causal_relationships']
        edges_df = pd.read_csv(edges_path)
        print(f"✅ Loaded {len(edges_df)} causal relationships")
        
        return df, edges_df
    
    def build_graph(self, df, edges_df):
        """Build causal graph structure"""
        print("\n🔗 BUILDING CAUSAL GRAPH")
        print("="*70)
        
        feature_names = list(df.columns)
        self.feature_names = feature_names
        feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        print(f"✅ Created feature mapping for {len(feature_names)} features")
        print(f"   Features: {feature_names}")
        
        edge_list = []
        edge_weights = []
        skipped = []
        
        for _, row in edges_df.iterrows():
            cause = row['cause']
            effect = row['effect']
            
            # Handle name variations (e.g., weather_severity vs weather_severity_index)
            if cause not in feature_to_idx:
                # Try to find partial match
                matches = [f for f in feature_names if cause in f or f in cause]
                if matches:
                    cause = matches[0]
                    print(f"   ⚠️ Mapped '{row['cause']}' → '{cause}'")
            
            if effect not in feature_to_idx:
                # Try to find partial match
                matches = [f for f in feature_names if effect in f or f in effect]
                if matches:
                    effect = matches[0]
                    print(f"   ⚠️ Mapped '{row['effect']}' → '{effect}'")
            
            # Skip if still not found
            if cause not in feature_to_idx or effect not in feature_to_idx:
                skipped.append((row['cause'], row['effect']))
                continue
            
            cause_idx = feature_to_idx[cause]
            effect_idx = feature_to_idx[effect]
            strength = row.get('strength', 1.0)
            
            edge_list.append([cause_idx, effect_idx])
            edge_weights.append(strength)
            
            # Bidirectional edges
            if row.get('type') == 'bidirectional':
                edge_list.append([effect_idx, cause_idx])
                edge_weights.append(strength)
        
        if skipped:
            print(f"\n   ⚠️ Skipped {len(skipped)} edges (features not in data):")
            for cause, effect in skipped[:5]:  # Show first 5
                print(f"      {cause} → {effect}")
            if len(skipped) > 5:
                print(f"      ... and {len(skipped) - 5} more")
        
        # Create tensors
        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        else:
            # Fallback: self-loops
            num_features = len(feature_names)
            edge_index = torch.tensor([[i, i] for i in range(num_features)], 
                                    dtype=torch.long).t().contiguous()
            edge_attr = torch.ones((num_features, 1), dtype=torch.float)
            print(f"   ⚠️ No valid edges found - using self-loops")
        
        print(f"\n✅ Graph structure:")
        print(f"   Nodes: {len(feature_names)}")
        print(f"   Edges: {edge_index.shape[1]}")
        print(f"   Edge weights shape: {edge_attr.shape}")
        
        return edge_index, edge_attr, feature_names

    
    def prepare_data(self):
        """Prepare train/val/test splits"""
        print("\n🎲 PREPARING DATA SPLITS")
        print("="*70)
        
        df, edges_df = self.load_data()
        
        target_col = 'Accident Severity'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found!")
        
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].values
        y = df[target_col].values
        
        print(f"✅ Features: {X.shape}")
        print(f"✅ Target: {y.shape}")
        print(f"✅ Classes: {np.bincount(y.astype(int))}")
        
        # Build graph
        edge_index, edge_attr, feature_names = self.build_graph(
            df[feature_cols], edges_df
        )
        
        # Split data
        train_size = self.config['data']['train_split']
        val_size = self.config['data']['val_split']
        random_seed = self.config['data']['random_seed']
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1-train_size), random_state=random_seed, stratify=y
        )
        
        val_ratio = val_size / (val_size + self.config['data']['test_split'])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1-val_ratio), 
            random_state=random_seed, stratify=y_temp
        )
        
        print(f"\n✅ Splits:")
        print(f"   Train: {len(y_train)} ({len(y_train)/len(y)*100:.1f}%)")
        print(f"   Val:   {len(y_val)} ({len(y_val)/len(y)*100:.1f}%)")
        print(f"   Test:  {len(y_test)} ({len(y_test)/len(y)*100:.1f}%)")
        
        # Normalize
        print(f"\n🔄 Normalizing features...")
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        # Create PyG Data objects
        train_data = self._create_data(X_train, y_train, edge_index, edge_attr)
        val_data = self._create_data(X_val, y_val, edge_index, edge_attr)
        test_data = self._create_data(X_test, y_test, edge_index, edge_attr)
        
        print("\n" + "="*70)
        print("✅ DATA PREPARATION COMPLETE")
        print("="*70 + "\n")
        
        return train_data, val_data, test_data, feature_names
    
    def _create_data(self, X, y, edge_index, edge_attr):
        """Create PyTorch Geometric Data object"""
        return Data(
            x=torch.tensor(X, dtype=torch.float),
            y=torch.tensor(y, dtype=torch.long),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=X.shape[1]
        )
    
    def save_scaler(self, path):
        """Save fitted scaler"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler saved to {path}")
    
    def load_scaler(self, path):
        """Load fitted scaler"""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✅ Scaler loaded from {path}")
