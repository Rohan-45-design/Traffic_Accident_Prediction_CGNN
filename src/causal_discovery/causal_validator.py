"""
Causal Relationship Validator
============================
Validates discovered causal relationships using statistical methods
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings

class CausalValidator:
    """Validates causal relationships discovered by PC algorithm"""
    
    def __init__(self, bootstrap_samples=100, confidence_level=0.95):
        """
        Initialize validator
        
        Parameters:
        -----------
        bootstrap_samples : int
            Number of bootstrap samples for validation
        confidence_level : float
            Confidence level for validation intervals
        """
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def validate_relationships(self, data, relationships):
        """
        Validate discovered causal relationships
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Original data used for causal discovery
        relationships : list
            List of discovered causal relationships
            
        Returns:
        --------
        validation_results : dict
            Validation results and statistics
        """
        print("📊 Validating causal relationships...")
        
        validated_relationships = []
        validation_stats = {
            'total_relationships': len(relationships),
            'validated_count': 0,
            'rejected_count': 0,
            'weak_evidence': 0,
            'bootstrap_stability': []
        }
        
        for i, relationship in enumerate(relationships):
            print(f"   Validating {i+1}/{len(relationships)}: {relationship['cause']} -> {relationship['effect']}")
            
            validation_result = self._validate_single_relationship(data, relationship)
            
            if validation_result['is_valid']:
                validated_relationships.append({
                    **relationship,
                    'validation': validation_result
                })
                validation_stats['validated_count'] += 1
                
                if validation_result['strength'] < 0.3:
                    validation_stats['weak_evidence'] += 1
            else:
                validation_stats['rejected_count'] += 1
        
        # Bootstrap stability analysis
        if self.bootstrap_samples > 0:
            stability_results = self._bootstrap_stability(data, relationships)
            validation_stats['bootstrap_stability'] = stability_results
        
        print(f"✅ Validation complete:")
        print(f"   - Total relationships: {validation_stats['total_relationships']}")
        print(f"   - Validated: {validation_stats['validated_count']}")
        print(f"   - Rejected: {validation_stats['rejected_count']}")
        print(f"   - Weak evidence: {validation_stats['weak_evidence']}")
        
        return {
            'validated_relationships': validated_relationships,
            'validation_statistics': validation_stats,
            'validated_count': validation_stats['validated_count']
        }
    
    def _validate_single_relationship(self, data, relationship):
        """Validate a single causal relationship"""
        cause = relationship['cause']
        effect = relationship['effect']
        
        try:
            # Test 1: Correlation strength
            correlation = data[cause].corr(data[effect])
            corr_strength = abs(correlation)
            
            # Test 2: Statistical significance
            corr_stat, p_value = stats.pearsonr(data[cause], data[effect])
            is_significant = p_value < self.alpha
            
            # Test 3: Effect size (practical significance)
            effect_size = self._calculate_effect_size(data[cause], data[effect])
            
            # Test 4: Direction consistency (basic check)
            direction_score = self._check_direction_consistency(data, cause, effect)
            
            # Overall validation decision
            is_valid = (
                is_significant and 
                corr_strength > 0.1 and  # Minimum correlation threshold
                effect_size > 0.1        # Minimum effect size
            )
            
            return {
                'is_valid': is_valid,
                'correlation': float(correlation),
                'p_value': float(p_value),
                'effect_size': float(effect_size),
                'direction_score': float(direction_score),
                'strength': float(corr_strength),
                'significance_level': self.alpha
            }
            
        except Exception as e:
            print(f"⚠️ Validation error for {cause} -> {effect}: {e}")
            return {
                'is_valid': False,
                'error': str(e),
                'correlation': 0.0,
                'p_value': 1.0,
                'effect_size': 0.0,
                'direction_score': 0.0,
                'strength': 0.0
            }
    
    def _calculate_effect_size(self, cause_data, effect_data):
        """Calculate effect size (Cohen's d equivalent for continuous variables)"""
        try:
            # For continuous variables, use correlation-based effect size
            correlation = cause_data.corr(effect_data)
            # Convert correlation to Cohen's d equivalent
            effect_size = 2 * abs(correlation) / np.sqrt(1 - correlation**2) if abs(correlation) < 0.99 else 2.0
            return min(effect_size, 2.0)  # Cap at reasonable maximum
        except:
            return 0.0
    
    def _check_direction_consistency(self, data, cause, effect):
        """Basic check for causal direction consistency"""
        try:
            # Simple heuristic: temporal precedence proxy
            # Check if cause values tend to precede effect changes
            
            # Calculate rolling correlation to see if relationship is stable
            window_size = min(100, len(data) // 10)
            if window_size < 10:
                return 0.5  # Neutral score for small datasets
            
            rolling_corr = data[cause].rolling(window_size).corr(data[effect].shift(-1))
            forward_corr = rolling_corr.mean()
            
            rolling_corr_backward = data[cause].rolling(window_size).corr(data[effect].shift(1))
            backward_corr = rolling_corr_backward.mean()
            
            if np.isnan(forward_corr) or np.isnan(backward_corr):
                return 0.5
            
            # Direction score: higher if forward correlation > backward correlation
            direction_score = (abs(forward_corr) - abs(backward_corr) + 1) / 2
            return np.clip(direction_score, 0, 1)
            
        except:
            return 0.5  # Neutral score if calculation fails
    
    def _bootstrap_stability(self, data, relationships):
        """Assess stability of causal relationships using bootstrap"""
        print("🔄 Performing bootstrap stability analysis...")
        
        stability_results = []
        n_samples = len(data)
        
        for relationship in relationships[:min(5, len(relationships))]:  # Limit to first 5 for speed
            cause = relationship['cause']
            effect = relationship['effect']
            
            bootstrap_correlations = []
            
            for _ in range(self.bootstrap_samples):
                # Bootstrap sample
                bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
                bootstrap_data = data.iloc[bootstrap_indices]
                
                try:
                    corr = bootstrap_data[cause].corr(bootstrap_data[effect])
                    if not np.isnan(corr):
                        bootstrap_correlations.append(corr)
                except:
                    continue
            
            if bootstrap_correlations:
                stability_score = np.std(bootstrap_correlations) / (np.mean(np.abs(bootstrap_correlations)) + 1e-6)
                stability_results.append({
                    'relationship': f"{cause} -> {effect}",
                    'stability_score': float(1 / (1 + stability_score)),  # Higher is more stable
                    'correlation_mean': float(np.mean(bootstrap_correlations)),
                    'correlation_std': float(np.std(bootstrap_correlations))
                })
        
        return stability_results
