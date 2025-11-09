"""
PC Algorithm Implementation for Causal Discovery
===============================================
Implements the Peter-Clark (PC) algorithm for discovering causal relationships
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import warnings

class PCAlgorithm:
    """PC Algorithm for causal structure learning"""
    
    def __init__(self, alpha=0.05, max_cond_vars=3, method='fisherz'):
        """
        Initialize PC Algorithm
        
        Parameters:
        -----------
        alpha : float
            Significance level for independence tests
        max_cond_vars : int
            Maximum number of conditioning variables
        method : str
            Independence test method ('fisherz', 'chi2', 'pearson')
        """
        self.alpha = alpha
        self.max_cond_vars = max_cond_vars
        self.method = method
        self.independence_cache = {}
        
    def discover_causal_structure(self, data):
        """
        Discover causal structure using PC algorithm
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data with variables as columns
            
        Returns:
        --------
        adjacency_matrix : numpy.array
            Adjacency matrix representing causal relationships
        relationships : list
            List of discovered causal relationships
        """
        print("🧠 Starting PC algorithm causal discovery...")
        
        variables = list(data.columns)
        n_vars = len(variables)
        
        # Initialize complete graph (all variables connected)
        adjacency = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        
        print(f"📊 Variables: {n_vars}")
        print(f"🔍 Initial connections: {np.sum(adjacency)}")
        
        # Phase 1: Skeleton discovery (remove edges based on independence)
        adjacency = self._skeleton_discovery(data, adjacency, variables)
        
        # Phase 2: Edge orientation (determine causal directions)
        oriented_graph = self._orient_edges(data, adjacency, variables)
        
        # Extract relationships
        relationships = self._extract_relationships(oriented_graph, variables)
        
        print(f"✅ Causal discovery complete: {len(relationships)} relationships found")
        
        return oriented_graph, relationships
    
    def _skeleton_discovery(self, data, adjacency, variables):
        """Phase 1: Remove edges based on conditional independence"""
        print("🔍 Phase 1: Skeleton discovery...")
        
        n_vars = len(variables)
        removed_edges = 0
        
        # Test independence with increasing conditioning set sizes
        for cond_size in range(self.max_cond_vars + 1):
            print(f"   Testing with {cond_size} conditioning variables...")
            
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if adjacency[i, j] == 0:  # Already removed
                        continue
                    
                    # Get potential conditioning variables
                    potential_cond = [k for k in range(n_vars) if k != i and k != j and adjacency[i, k] == 1]
                    
                    if len(potential_cond) >= cond_size:
                        # Test all combinations of conditioning variables
                        for cond_vars in combinations(potential_cond, cond_size):
                            if self._test_independence(data, i, j, list(cond_vars), variables):
                                adjacency[i, j] = adjacency[j, i] = 0  # Remove edge
                                removed_edges += 1
                                break
        
        remaining_edges = np.sum(adjacency) // 2
        print(f"   Removed {removed_edges} edges, {remaining_edges} remain")
        
        return adjacency
    
    def _orient_edges(self, data, adjacency, variables):
        """Phase 2: Orient edges to determine causal directions"""
        print("🔍 Phase 2: Edge orientation...")
        
        # Create directed adjacency matrix (0: no edge, 1: X->Y, -1: X<-Y)
        oriented = adjacency.copy()
        
        # Apply orientation rules
        oriented = self._apply_orientation_rules(oriented, variables)
        
        return oriented
    
    def _apply_orientation_rules(self, adjacency, variables):
        """Apply PC algorithm orientation rules"""
        n_vars = len(variables)
        oriented = adjacency.copy()
        
        # Rule 1: Orient v-structures (X -> Z <- Y where X and Y not connected)
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j and oriented[i, j] != 0:
                    for k in range(n_vars):
                        if k != i and k != j and oriented[k, j] != 0 and oriented[i, k] == 0:
                            # Found v-structure: i -> j <- k
                            if oriented[i, j] == 1:  # Undirected
                                oriented[i, j] = 1   # i -> j
                                oriented[j, i] = -1  # j <- i
                            if oriented[k, j] == 1:  # Undirected
                                oriented[k, j] = 1   # k -> j  
                                oriented[j, k] = -1  # j <- k
        
        return oriented
    
    def _test_independence(self, data, var1_idx, var2_idx, cond_vars, variables):
        """Test conditional independence between two variables"""
        var1_name = variables[var1_idx]
        var2_name = variables[var2_idx]
        cond_names = [variables[i] for i in cond_vars]
        
        # Create cache key
        cache_key = tuple(sorted([var1_name, var2_name]) + sorted(cond_names))
        if cache_key in self.independence_cache:
            return self.independence_cache[cache_key]
        
        try:
            if self.method == 'fisherz':
                result = self._fisherz_test(data, var1_name, var2_name, cond_names)
            elif self.method == 'pearson':
                result = self._pearson_test(data, var1_name, var2_name, cond_names)
            else:
                result = self._fisherz_test(data, var1_name, var2_name, cond_names)
            
            self.independence_cache[cache_key] = result
            return result
            
        except Exception as e:
            print(f"⚠️ Independence test failed for {var1_name}-{var2_name}: {e}")
            return False
    
    def _fisherz_test(self, data, var1, var2, cond_vars):
        """Fisher's Z test for conditional independence"""
        n = len(data)
        
        if len(cond_vars) == 0:
            # Unconditional correlation
            corr = data[var1].corr(data[var2])
        else:
            # Partial correlation
            corr = self._partial_correlation(data, var1, var2, cond_vars)
        
        if abs(corr) > 0.99:  # Avoid numerical issues
            return abs(corr) < 0.99
        
        # Fisher's Z transformation
        z_score = 0.5 * np.log((1 + abs(corr)) / (1 - abs(corr)))
        test_stat = z_score * np.sqrt(n - len(cond_vars) - 3)
        p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        
        return p_value > self.alpha
    
    def _pearson_test(self, data, var1, var2, cond_vars):
        """Pearson correlation test"""
        if len(cond_vars) == 0:
            corr, p_value = stats.pearsonr(data[var1], data[var2])
        else:
            corr = self._partial_correlation(data, var1, var2, cond_vars)
            # Approximate p-value for partial correlation
            n = len(data)
            df = n - len(cond_vars) - 2
            t_stat = corr * np.sqrt(df / (1 - corr**2)) if abs(corr) < 0.99 else 10
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        return p_value > self.alpha
    
    def _partial_correlation(self, data, var1, var2, cond_vars):
        """Calculate partial correlation"""
        if len(cond_vars) == 0:
            return data[var1].corr(data[var2])
        
        # Create regression data
        all_vars = [var1, var2] + cond_vars
        subset_data = data[all_vars].dropna()
        
        if len(subset_data) < 10:  # Too few samples
            return 0.0
        
        try:
            # Calculate correlation matrix
            corr_matrix = subset_data.corr().values
            
            # Extract relevant parts
            n_cond = len(cond_vars)
            
            # Partial correlation formula using matrix operations
            if n_cond > 0:
                # Precision matrix (inverse correlation matrix)
                precision = np.linalg.inv(corr_matrix)
                partial_corr = -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])
            else:
                partial_corr = corr_matrix[0, 1]
            
            return partial_corr
            
        except (np.linalg.LinAlgError, ZeroDivisionError):
            return 0.0
    
    def _extract_relationships(self, adjacency, variables):
        """Extract causal relationships from adjacency matrix"""
        relationships = []
        n_vars = len(variables)
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j and adjacency[i, j] > 0:  # Edge from i to j
                    relationships.append({
                        'cause': variables[i],
                        'effect': variables[j],
                        'strength': float(adjacency[i, j]),
                        'type': 'causal' if adjacency[j, i] <= 0 else 'bidirectional'
                    })
        
        return relationships
