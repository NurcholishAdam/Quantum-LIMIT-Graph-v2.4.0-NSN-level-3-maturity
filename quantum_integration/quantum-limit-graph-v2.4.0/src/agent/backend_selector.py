# -*- coding: utf-8 -*-
"""
Backend Selector Module
Intelligent backend selection based on edit characteristics
"""
import numpy as np
from typing import Dict, List, Optional


class PerformancePredictor:
    """Predict backend performance for given edit"""
    
    def __init__(self):
        """Initialize performance predictor"""
        # Performance profiles for each backend
        self.profiles = {
            'russian': {
                'cyrillic_boost': 0.05,
                'base_success': 0.873,
                'base_latency': 85,
                'domain_strengths': {'scientific': 0.03, 'math': 0.02}
            },
            'ibm': {
                'cyrillic_boost': 0.0,
                'base_success': 0.891,
                'base_latency': 92,
                'domain_strengths': {'code': 0.03, 'text': 0.02}
            }
        }
    
    def predict_performance(
        self,
        backend: str,
        edit_features: Dict
    ) -> Dict[str, float]:
        """
        Predict performance for backend on edit
        
        Args:
            backend: Backend name
            edit_features: Edit characteristics
            
        Returns:
            Predicted performance metrics
        """
        profile = self.profiles.get(backend, self.profiles['ibm'])
        
        # Base success rate
        success_rate = profile['base_success']
        
        # Language boost
        if edit_features.get('lang') == 'ru' and backend == 'russian':
            success_rate += profile['cyrillic_boost']
        
        # Domain boost
        domain = edit_features.get('domain', 'text')
        if domain in profile['domain_strengths']:
            success_rate += profile['domain_strengths'][domain]
        
        # Latency
        latency = profile['base_latency']
        
        # Add noise
        success_rate += np.random.normal(0, 0.01)
        latency += np.random.normal(0, 5)
        
        return {
            'predicted_success_rate': min(1.0, max(0.0, success_rate)),
            'predicted_latency_ms': max(10, latency),
            'confidence': 0.85
        }


class BackendSelector:
    """Select optimal backend for edit"""
    
    def __init__(
        self,
        backends: List[str] = None,
        selection_strategy: str = 'performance'
    ):
        """
        Initialize backend selector
        
        Args:
            backends: Available backends
            selection_strategy: Strategy ('performance', 'latency', 'balanced')
        """
        self.backends = backends or ['russian', 'ibm']
        self.selection_strategy = selection_strategy
        self.predictor = PerformancePredictor()
        
        self.selection_history = []
    
    def select_backend(
        self,
        edit: Dict,
        constraints: Optional[Dict] = None
    ) -> str:
        """
        Select optimal backend for edit
        
        Args:
            edit: Edit to process
            constraints: Optional constraints (max_latency, min_success_rate)
            
        Returns:
            Selected backend name
        """
        constraints = constraints or {}
        
        # Extract edit features
        features = {
            'lang': edit.get('lang', 'en'),
            'domain': edit.get('domain', 'text'),
            'complexity': edit.get('complexity', 'medium')
        }
        
        # Predict performance for each backend
        predictions = {}
        for backend in self.backends:
            pred = self.predictor.predict_performance(backend, features)
            predictions[backend] = pred
        
        # Select based on strategy
        if self.selection_strategy == 'performance':
            selected = max(
                predictions.items(),
                key=lambda x: x[1]['predicted_success_rate']
            )[0]
        elif self.selection_strategy == 'latency':
            selected = min(
                predictions.items(),
                key=lambda x: x[1]['predicted_latency_ms']
            )[0]
        else:  # balanced
            selected = max(
                predictions.items(),
                key=lambda x: x[1]['predicted_success_rate'] / (x[1]['predicted_latency_ms'] / 100)
            )[0]
        
        # Apply constraints
        if constraints:
            max_latency = constraints.get('max_latency_ms')
            min_success = constraints.get('min_success_rate')
            
            if max_latency and predictions[selected]['predicted_latency_ms'] > max_latency:
                # Find alternative
                for backend, pred in predictions.items():
                    if pred['predicted_latency_ms'] <= max_latency:
                        selected = backend
                        break
            
            if min_success and predictions[selected]['predicted_success_rate'] < min_success:
                # Find alternative
                for backend, pred in predictions.items():
                    if pred['predicted_success_rate'] >= min_success:
                        selected = backend
                        break
        
        # Record selection
        self.selection_history.append({
            'edit_id': edit.get('id', 'unknown'),
            'selected_backend': selected,
            'predictions': predictions
        })
        
        return selected
    
    def batch_select(
        self,
        edits: List[Dict],
        constraints: Optional[Dict] = None
    ) -> Dict[str, List[Dict]]:
        """
        Select backends for batch of edits
        
        Args:
            edits: List of edits
            constraints: Optional constraints
            
        Returns:
            Dict mapping backend names to edit lists
        """
        backend_assignments = {backend: [] for backend in self.backends}
        
        for edit in edits:
            selected = self.select_backend(edit, constraints)
            backend_assignments[selected].append(edit)
        
        return backend_assignments
    
    def get_selection_stats(self) -> Dict:
        """Get selection statistics"""
        if not self.selection_history:
            return {}
        
        backend_counts = {backend: 0 for backend in self.backends}
        for selection in self.selection_history:
            backend_counts[selection['selected_backend']] += 1
        
        total = len(self.selection_history)
        return {
            'total_selections': total,
            'backend_distribution': {
                backend: count / total
                for backend, count in backend_counts.items()
            }
        }


# Convenience function
def select_best_backend(edit: Dict, backends: List[str] = None) -> str:
    """Quick backend selection"""
    selector = BackendSelector(backends=backends)
    return selector.select_backend(edit)
