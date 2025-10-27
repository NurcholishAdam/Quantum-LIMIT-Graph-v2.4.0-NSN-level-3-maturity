# -*- coding: utf-8 -*-
"""
Leaderboard Generator Module
Generates performance leaderboards for backend comparison
"""
import json
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class LeaderboardEntry:
    """Single leaderboard entry"""
    rank: int
    backend: str
    score: float
    metrics: Dict[str, float]


class RankingEngine:
    """Engine for ranking backends"""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize ranking engine
        
        Args:
            weights: Metric weights for scoring
        """
        self.weights = weights or {
            'edit_success_rate': 0.35,
            'correction_efficiency': 0.25,
            'circuit_fidelity': 0.20,
            'throughput_edits_per_sec': 0.10,
            'hallucination_rate': -0.10  # Negative weight
        }
    
    def calculate_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted score"""
        score = 0.0
        for metric, weight in self.weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
        return max(0.0, min(100.0, score * 100))


class LeaderboardGenerator:
    """Generate performance leaderboards"""
    
    def __init__(self):
        """Initialize leaderboard generator"""
        self.ranking_engine = RankingEngine()
        self.leaderboard = []
    
    def generate_leaderboard(
        self,
        backend_results: Dict
    ) -> List[LeaderboardEntry]:
        """
        Generate leaderboard from backend results
        
        Args:
            backend_results: Dict of backend metrics
            
        Returns:
            Sorted list of leaderboard entries
        """
        entries = []
        
        for backend, metrics in backend_results.items():
            # Convert metrics to dict
            if hasattr(metrics, '__dict__'):
                metrics_dict = metrics.__dict__
            else:
                metrics_dict = metrics
            
            # Calculate score
            score = self.ranking_engine.calculate_score(metrics_dict)
            
            entries.append({
                'backend': backend,
                'score': score,
                'metrics': metrics_dict
            })
        
        # Sort by score
        entries.sort(key=lambda x: x['score'], reverse=True)
        
        # Add ranks
        leaderboard = []
        for rank, entry in enumerate(entries, 1):
            leaderboard.append(LeaderboardEntry(
                rank=rank,
                backend=entry['backend'],
                score=entry['score'],
                metrics=entry['metrics']
            ))
        
        self.leaderboard = leaderboard
        return leaderboard
    
    def display_leaderboard(self, leaderboard: List[LeaderboardEntry] = None):
        """Display leaderboard in console"""
        leaderboard = leaderboard or self.leaderboard
        
        if not leaderboard:
            print("⚠️  No leaderboard data available.")
            return
        
        print("\n" + "=" * 70)
        print("🏆 QUANTUM BACKEND LEADERBOARD")
        print("=" * 70)
        
        for entry in leaderboard:
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(entry.rank, "  ")
            print(f"\n{medal} Rank #{entry.rank}: {entry.backend.upper()}")
            print(f"   Score: {entry.score:.1f}/100")
            print(f"   Success Rate: {entry.metrics.get('edit_success_rate', 0)*100:.1f}%")
            print(f"   Fidelity: {entry.metrics.get('circuit_fidelity', 0):.3f}")
        
        print("\n" + "=" * 70)
    
    def export_leaderboard(
        self,
        filepath: str = 'leaderboard.json',
        leaderboard: List[LeaderboardEntry] = None
    ):
        """Export leaderboard to JSON"""
        leaderboard = leaderboard or self.leaderboard
        
        export_data = [
            {
                'rank': entry.rank,
                'backend': entry.backend,
                'score': entry.score,
                'metrics': entry.metrics
            }
            for entry in leaderboard
        ]
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Leaderboard exported: {filepath}")
