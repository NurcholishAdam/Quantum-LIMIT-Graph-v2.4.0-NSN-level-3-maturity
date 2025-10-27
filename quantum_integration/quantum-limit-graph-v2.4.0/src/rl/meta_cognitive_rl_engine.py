# -*- coding: utf-8 -*-
"""
Stage 2: Meta-Cognitive RL for Contributor Feedback
Makes leaderboard and feedback loop adaptive through personalized RL
"""
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContributorProfile:
    """Profile for a contributor"""
    contributor_id: str
    submission_history: List[Dict] = field(default_factory=list)
    skill_vector: np.ndarray = field(default_factory=lambda: np.zeros(10))
    preferred_domains: List[str] = field(default_factory=list)
    avg_edit_quality: float = 0.0
    engagement_score: float = 0.0
    badges: List[str] = field(default_factory=list)


@dataclass
class RLState:
    """RL state for contributor feedback"""
    contributor_profile: ContributorProfile
    current_edit: Dict
    leaderboard_position: int
    recent_feedback: List[float]
    backend_performance: Dict[str, float]


@dataclass
class RLAction:
    """RL action for feedback"""
    recommended_edit_type: str
    recommended_backend: str
    suggested_difficulty: str
    badge_assignment: Optional[str]
    personalized_tip: str


class QuantumPolicyOptimizer:
    """
    Quantum-enhanced policy optimizer for RL
    Uses quantum circuits to optimize policy updates
    """
    
    def __init__(self, num_qubits: int = 6):
        """
        Initialize quantum policy optimizer
        
        Args:
            num_qubits: Number of qubits for policy circuit
        """
        self.num_qubits = num_qubits
        self.backend = AerSimulator()
        self.policy_params = np.random.randn(num_qubits * 3)
        
        logger.info(f"Initialized QuantumPolicyOptimizer with {num_qubits} qubits")
    
    def build_policy_circuit(self, state_vector: np.ndarray) -> QuantumCircuit:
        """
        Build quantum circuit for policy
        
        Args:
            state_vector: Current state representation
            
        Returns:
            Quantum circuit
        """
        qc = QuantumCircuit(self.num_qubits)
        
        # Encode state into circuit
        for i in range(min(self.num_qubits, len(state_vector))):
            angle = state_vector[i] * np.pi
            qc.ry(angle, i)
        
        # Apply parameterized gates
        for i in range(self.num_qubits):
            qc.rz(self.policy_params[i], i)
        
        # Entanglement layer
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        
        # Second rotation layer
        for i in range(self.num_qubits):
            qc.ry(self.policy_params[self.num_qubits + i], i)
        
        return qc
    
    def compute_policy_gradient(self, 
                                state: np.ndarray, 
                                reward: float) -> np.ndarray:
        """
        Compute quantum policy gradient
        
        Args:
            state: Current state
            reward: Received reward
            
        Returns:
            Policy gradient
        """
        # Build policy circuit
        qc = self.build_policy_circuit(state)
        qc.measure_all()
        
        # Execute
        job = self.backend.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Compute expectation value
        expectation = 0.0
        for bitstring, count in counts.items():
            value = int(bitstring, 2) / (2**self.num_qubits)
            expectation += value * (count / 1000)
        
        # Gradient approximation
        gradient = np.zeros_like(self.policy_params)
        epsilon = 0.01
        
        for i in range(len(self.policy_params)):
            # Finite difference
            self.policy_params[i] += epsilon
            qc_plus = self.build_policy_circuit(state)
            qc_plus.measure_all()
            
            job_plus = self.backend.run(qc_plus, shots=100)
            result_plus = job_plus.result()
            counts_plus = result_plus.get_counts()
            
            exp_plus = sum(
                int(bs, 2) / (2**self.num_qubits) * (c / 100)
                for bs, c in counts_plus.items()
            )
            
            gradient[i] = (exp_plus - expectation) / epsilon * reward
            self.policy_params[i] -= epsilon
        
        return gradient
    
    def update_policy(self, gradient: np.ndarray, learning_rate: float = 0.01):
        """
        Update policy parameters
        
        Args:
            gradient: Policy gradient
            learning_rate: Learning rate
        """
        self.policy_params += learning_rate * gradient
        logger.info(f"Updated policy parameters: norm={np.linalg.norm(gradient):.4f}")


class MetaCognitiveRLEngine:
    """
    Meta-cognitive RL engine for adaptive contributor feedback
    Personalizes recommendations and optimizes engagement
    """
    
    def __init__(self, num_contributors: int = 100):
        """
        Initialize meta-cognitive RL engine
        
        Args:
            num_contributors: Expected number of contributors
        """
        self.contributor_profiles: Dict[str, ContributorProfile] = {}
        self.policy_optimizer = QuantumPolicyOptimizer()
        self.reward_history: List[float] = []
        
        logger.info(f"Initialized MetaCognitiveRLEngine for {num_contributors} contributors")
    
    def get_or_create_profile(self, contributor_id: str) -> ContributorProfile:
        """Get or create contributor profile"""
        if contributor_id not in self.contributor_profiles:
            self.contributor_profiles[contributor_id] = ContributorProfile(
                contributor_id=contributor_id
            )
        return self.contributor_profiles[contributor_id]
    
    def encode_state(self, state: RLState) -> np.ndarray:
        """
        Encode RL state into vector
        
        Args:
            state: RL state
            
        Returns:
            State vector
        """
        # Combine various state components
        state_vector = np.concatenate([
            state.contributor_profile.skill_vector[:5],
            [state.contributor_profile.avg_edit_quality],
            [state.contributor_profile.engagement_score],
            [state.leaderboard_position / 100.0],
            state.recent_feedback[-3:] if len(state.recent_feedback) >= 3 else [0, 0, 0]
        ])
        
        return state_vector
    
    def select_action(self, state: RLState) -> RLAction:
        """
        Select action using quantum policy
        
        Args:
            state: Current RL state
            
        Returns:
            Selected action
        """
        # Encode state
        state_vector = self.encode_state(state)
        
        # Build and execute policy circuit
        qc = self.policy_optimizer.build_policy_circuit(state_vector)
        qc.measure_all()
        
        job = self.policy_optimizer.backend.run(qc, shots=100)
        result = job.result()
        counts = result.get_counts()
        
        # Sample action from distribution
        most_common = max(counts.items(), key=lambda x: x[1])[0]
        action_code = int(most_common, 2)
        
        # Decode action
        edit_types = ['factual', 'grammatical', 'stylistic', 'structural']
        backends = ['russian', 'ibm', 'google', 'simulator']
        difficulties = ['easy', 'medium', 'hard', 'expert']
        
        action = RLAction(
            recommended_edit_type=edit_types[action_code % len(edit_types)],
            recommended_backend=backends[(action_code // 4) % len(backends)],
            suggested_difficulty=difficulties[(action_code // 16) % len(difficulties)],
            badge_assignment=self._determine_badge(state),
            personalized_tip=self._generate_tip(state)
        )
        
        logger.info(f"Selected action for {state.contributor_profile.contributor_id}: "
                   f"{action.recommended_edit_type} on {action.recommended_backend}")
        
        return action
    
    def compute_reward(self, 
                      state: RLState, 
                      action: RLAction, 
                      outcome: Dict) -> float:
        """
        Compute reward for state-action-outcome
        
        Args:
            state: State
            action: Action taken
            outcome: Outcome of action
            
        Returns:
            Reward value
        """
        # Base reward from edit quality
        quality_reward = outcome.get('edit_quality', 0.5)
        
        # Engagement reward
        engagement_reward = 0.0
        if outcome.get('contributor_returned', False):
            engagement_reward = 0.3
        
        # Difficulty match reward
        difficulty_match = 0.0
        if outcome.get('difficulty_appropriate', False):
            difficulty_match = 0.2
        
        # Backend performance reward
        backend_reward = outcome.get('backend_success', 0.5) * 0.2
        
        # Total reward
        reward = (
            0.4 * quality_reward +
            0.3 * engagement_reward +
            0.2 * difficulty_match +
            0.1 * backend_reward
        )
        
        self.reward_history.append(reward)
        
        return reward
    
    def update_from_feedback(self, 
                            contributor_id: str,
                            state: RLState,
                            action: RLAction,
                            outcome: Dict):
        """
        Update RL engine from feedback
        
        Args:
            contributor_id: Contributor ID
            state: State
            action: Action taken
            outcome: Outcome
        """
        # Compute reward
        reward = self.compute_reward(state, action, outcome)
        
        # Encode state
        state_vector = self.encode_state(state)
        
        # Compute policy gradient
        gradient = self.policy_optimizer.compute_policy_gradient(state_vector, reward)
        
        # Update policy
        self.policy_optimizer.update_policy(gradient)
        
        # Update contributor profile
        profile = self.get_or_create_profile(contributor_id)
        profile.submission_history.append({
            'state': state,
            'action': action,
            'outcome': outcome,
            'reward': reward
        })
        
        # Update skill vector
        profile.skill_vector = self._update_skill_vector(profile, outcome)
        
        # Update metrics
        profile.avg_edit_quality = np.mean([
            h['outcome'].get('edit_quality', 0.5)
            for h in profile.submission_history[-10:]
        ])
        
        profile.engagement_score = len(profile.submission_history) / 100.0
        
        logger.info(f"Updated profile for {contributor_id}: "
                   f"quality={profile.avg_edit_quality:.3f}, "
                   f"engagement={profile.engagement_score:.3f}")
    
    def _update_skill_vector(self, 
                            profile: ContributorProfile, 
                            outcome: Dict) -> np.ndarray:
        """Update contributor skill vector"""
        skill_vector = profile.skill_vector.copy()
        
        # Update based on outcome
        edit_type = outcome.get('edit_type', 'factual')
        quality = outcome.get('edit_quality', 0.5)
        
        # Map edit type to skill dimension
        type_to_dim = {
            'factual': 0,
            'grammatical': 1,
            'stylistic': 2,
            'structural': 3
        }
        
        dim = type_to_dim.get(edit_type, 0)
        
        # Update with learning rate
        learning_rate = 0.1
        skill_vector[dim] = (1 - learning_rate) * skill_vector[dim] + learning_rate * quality
        
        return skill_vector
    
    def _determine_badge(self, state: RLState) -> Optional[str]:
        """Determine if badge should be assigned"""
        profile = state.contributor_profile
        
        # Badge criteria
        if profile.avg_edit_quality > 0.9 and len(profile.submission_history) > 50:
            if 'expert' not in profile.badges:
                return 'expert'
        
        if profile.avg_edit_quality > 0.8 and len(profile.submission_history) > 20:
            if 'advanced' not in profile.badges:
                return 'advanced'
        
        if len(profile.submission_history) > 10:
            if 'contributor' not in profile.badges:
                return 'contributor'
        
        return None
    
    def _generate_tip(self, state: RLState) -> str:
        """Generate personalized tip"""
        profile = state.contributor_profile
        
        # Analyze weaknesses
        if profile.avg_edit_quality < 0.6:
            return "Focus on simpler edits to build confidence"
        
        if len(profile.submission_history) < 5:
            return "Try different edit types to discover your strengths"
        
        # Find weakest skill
        weakest_dim = np.argmin(profile.skill_vector[:4])
        skill_names = ['factual', 'grammatical', 'stylistic', 'structural']
        
        return f"Consider practicing {skill_names[weakest_dim]} edits"
    
    def get_personalized_recommendations(self, 
                                        contributor_id: str,
                                        available_edits: List[Dict]) -> List[Dict]:
        """
        Get personalized edit recommendations
        
        Args:
            contributor_id: Contributor ID
            available_edits: Available edits
            
        Returns:
            Ranked list of recommended edits
        """
        profile = self.get_or_create_profile(contributor_id)
        
        # Score each edit
        scored_edits = []
        
        for edit in available_edits:
            # Match to contributor skills
            edit_type = edit.get('type', 'factual')
            type_to_dim = {'factual': 0, 'grammatical': 1, 'stylistic': 2, 'structural': 3}
            dim = type_to_dim.get(edit_type, 0)
            
            skill_match = profile.skill_vector[dim]
            
            # Difficulty match
            difficulty = edit.get('difficulty', 'medium')
            difficulty_score = {
                'easy': 0.3,
                'medium': 0.6,
                'hard': 0.9,
                'expert': 1.0
            }.get(difficulty, 0.5)
            
            # Combine scores
            score = 0.6 * skill_match + 0.4 * difficulty_score
            
            scored_edits.append((score, edit))
        
        # Sort by score
        scored_edits.sort(reverse=True, key=lambda x: x[0])
        
        return [edit for score, edit in scored_edits[:10]]
    
    def get_engagement_metrics(self) -> Dict:
        """Get overall engagement metrics"""
        if not self.contributor_profiles:
            return {}
        
        profiles = list(self.contributor_profiles.values())
        
        return {
            'total_contributors': len(profiles),
            'avg_submissions': np.mean([len(p.submission_history) for p in profiles]),
            'avg_quality': np.mean([p.avg_edit_quality for p in profiles]),
            'avg_engagement': np.mean([p.engagement_score for p in profiles]),
            'total_badges_awarded': sum(len(p.badges) for p in profiles),
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0.0
        }


def demo_meta_cognitive_rl():
    """Demo meta-cognitive RL engine"""
    logger.info("=" * 80)
    logger.info("META-COGNITIVE RL ENGINE DEMO")
    logger.info("=" * 80)
    
    # Initialize engine
    engine = MetaCognitiveRLEngine()
    
    # Simulate contributor interactions
    contributor_id = "contributor_001"
    
    for episode in range(5):
        logger.info(f"\n--- Episode {episode + 1} ---")
        
        # Create state
        profile = engine.get_or_create_profile(contributor_id)
        state = RLState(
            contributor_profile=profile,
            current_edit={'type': 'factual', 'difficulty': 'medium'},
            leaderboard_position=50,
            recent_feedback=[0.7, 0.8, 0.75],
            backend_performance={'russian': 0.87, 'ibm': 0.89}
        )
        
        # Select action
        action = engine.select_action(state)
        logger.info(f"Action: {action.recommended_edit_type} on {action.recommended_backend}")
        logger.info(f"Tip: {action.personalized_tip}")
        
        # Simulate outcome
        outcome = {
            'edit_quality': np.random.uniform(0.6, 0.95),
            'contributor_returned': True,
            'difficulty_appropriate': True,
            'backend_success': 0.88,
            'edit_type': action.recommended_edit_type
        }
        
        # Update from feedback
        engine.update_from_feedback(contributor_id, state, action, outcome)
    
    # Get metrics
    metrics = engine.get_engagement_metrics()
    logger.info(f"\n--- Engagement Metrics ---")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")
    
    logger.info("\n" + "=" * 80)
    logger.info("DEMO COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    demo_meta_cognitive_rl() 
