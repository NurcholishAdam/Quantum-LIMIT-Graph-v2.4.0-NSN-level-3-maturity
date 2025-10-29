# -*- coding: utf-8 -*-
"""
Comprehensive tests for Level 3 maturity features
"""
import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from level_3_agent import Level3Agent, Level3Config
from src.core.quantum_transformer_core import QuantumTransformerCore, QuantumTransformerConfig
from src.rl.meta_cognitive_rl_engine import MetaCognitiveRLEngine
from src.perception.multimodal_quantum_processor import MultimodalQuantumProcessor
from src.orchestration.adaptive_backend_optimizer import AdaptiveBackendOptimizer
from src.memory.entangled_memory_store import EntangledMemoryStore
from src.domains.domain_module_factory import DomainModuleFactory


class TestQuantumTransformers:
    """Test Stage 1: Quantum Transformers"""
    
    def test_initialization(self):
        """Test quantum transformer initialization"""
        config = QuantumTransformerConfig(num_qubits=8, num_layers=4)
        core = QuantumTransformerCore(config)
        assert core.config.num_qubits == 8
        assert core.config.num_layers == 4
    
    def test_forward_pass(self):
        """Test quantum forward pass"""
        config = QuantumTransformerConfig(num_qubits=8)
        core = QuantumTransformerCore(config)
        
        input_vector = np.random.randn(128)
        result = core.quantum_forward_pass(input_vector, 'english')
        
        assert 'output_vector' in result
        assert 'coherence' in result
        assert 0 <= result['coherence'] <= 1
    
    def test_multilingual_inference(self):
        """Test multilingual inference"""
        config = QuantumTransformerConfig(num_qubits=8)
        core = QuantumTransformerCore(config)
        
        result = core.multilingual_inference(
            edit_text="Test edit",
            source_lang="english",
            target_langs=["chinese", "indonesian"]
        )
        
        assert 'source_language' in result
        assert 'target_results' in result
        assert len(result['target_results']) == 2


class TestMetaCognitiveRL:
    """Test Stage 2: Meta-Cognitive RL"""
    
    def test_initialization(self):
        """Test RL engine initialization"""
        engine = MetaCognitiveRLEngine()
        assert engine.contributor_profiles == {}
    
    def test_profile_creation(self):
        """Test contributor profile creation"""
        engine = MetaCognitiveRLEngine()
        profile = engine.get_or_create_profile('test_contributor')
        
        assert profile.contributor_id == 'test_contributor'
        assert len(profile.submission_history) == 0
    
    def test_action_selection(self):
        """Test action selection"""
        engine = MetaCognitiveRLEngine()
        profile = engine.get_or_create_profile('test_contributor')
        
        from src.rl.meta_cognitive_rl_engine import RLState
        state = RLState(
            contributor_profile=profile,
            current_edit={'type': 'factual'},
            leaderboard_position=50,
            recent_feedback=[0.8, 0.85],
            backend_performance={'russian': 0.87}
        )
        
        action = engine.select_action(state)
        assert action.recommended_edit_type in ['factual', 'grammatical', 'stylistic', 'structural']


class TestMultimodalPerception:
    """Test Stage 3: Multimodal Perception"""
    
    def test_initialization(self):
        """Test multimodal processor initialization"""
        processor = MultimodalQuantumProcessor()
        assert processor.ocr_engine is not None
        assert processor.scientific_simulator is not None
    
    def test_text_processing(self):
        """Test text processing"""
        processor = MultimodalQuantumProcessor()
        
        from src.perception.multimodal_quantum_processor import MultimodalInput
        input_data = MultimodalInput(text="Test text")
        
        result = processor.process_multimodal_input(input_data)
        assert 'text' in result
        assert result['text']['processed']
    
    def test_image_processing(self):
        """Test image processing"""
        processor = MultimodalQuantumProcessor()
        
        from src.perception.multimodal_quantum_processor import MultimodalInput
        input_data = MultimodalInput(image=np.random.rand(64, 64))
        
        result = processor.process_multimodal_input(input_data)
        assert 'image' in result
        assert 'confidence' in result['image']


class TestAdaptiveBackendOptimization:
    """Test Stage 4: Adaptive Backend Optimization"""
    
    def test_initialization(self):
        """Test backend optimizer initialization"""
        optimizer = AdaptiveBackendOptimizer()
        assert optimizer.router is not None
        assert optimizer.fallback_manager is not None
    
    def test_backend_selection(self):
        """Test backend selection"""
        optimizer = AdaptiveBackendOptimizer()
        
        edit_data = {'id': 'test_edit', 'type': 'factual'}
        result = optimizer.select_optimal_backend(edit_data)
        
        assert result.selected_backend is not None
        assert 0 <= result.confidence <= 1
        assert isinstance(result.fallback_backends, list)
    
    def test_execution_with_fallback(self):
        """Test execution with fallback"""
        optimizer = AdaptiveBackendOptimizer()
        
        edit_data = {'id': 'test_edit', 'content': 'test'}
        result = optimizer.execute_with_optimization(edit_data)
        
        assert 'success' in result
        assert 'backend' in result


class TestEntangledMemory:
    """Test Stage 5: Entangled Memory"""
    
    def test_initialization(self):
        """Test memory store initialization"""
        store = EntangledMemoryStore(max_shards=100)
        assert store.max_shards == 100
        assert len(store.shards) == 0
    
    def test_store_edit(self):
        """Test edit storage"""
        store = EntangledMemoryStore()
        
        shard = store.store_edit(
            edit_content="Test edit",
            language="english",
            rank=128
        )
        
        assert shard.shard_id is not None
        assert shard.language == "english"
        assert shard.rank == 128
        assert 0 <= shard.coherence <= 1
    
    def test_propagate_edit(self):
        """Test edit propagation"""
        store = EntangledMemoryStore()
        
        shard1 = store.store_edit("Test edit", "english", 128)
        shard2 = store.propagate_edit(shard1.shard_id, "chinese", 128)
        
        assert shard2.parent_shard_id == shard1.shard_id
        assert shard2.language == "chinese"
    
    def test_lineage_tracking(self):
        """Test lineage tracking"""
        store = EntangledMemoryStore()
        
        shard1 = store.store_edit("Test", "english", 128)
        shard2 = store.propagate_edit(shard1.shard_id, "chinese", 128)
        shard3 = store.propagate_edit(shard2.shard_id, "indonesian", 64)
        
        lineage = store.lineage_tracker.get_lineage(shard3.shard_id)
        assert lineage.depth == 2
        assert len(lineage.propagation_path) == 3


class TestDomainModules:
    """Test Stage 6: Domain Modules"""
    
    def test_factory_initialization(self):
        """Test domain factory initialization"""
        factory = DomainModuleFactory()
        assert factory.registered_domains == {}
    
    def test_create_default_modules(self):
        """Test default module creation"""
        factory = DomainModuleFactory()
        modules = factory.create_default_modules()
        
        assert 'legal' in modules
        assert 'medical' in modules
        assert 'scientific' in modules
    
    def test_legal_validation(self):
        """Test legal domain validation"""
        factory = DomainModuleFactory()
        modules = factory.create_default_modules()
        
        legal_module = modules['legal']
        result = legal_module.validate_edit(
            "In Smith v. Jones, the plaintiff argued",
            "en"
        )
        
        assert 'is_valid' in result
        assert 'legal_accuracy' in result
    
    def test_medical_validation(self):
        """Test medical domain validation"""
        factory = DomainModuleFactory()
        modules = factory.create_default_modules()
        
        medical_module = modules['medical']
        result = medical_module.validate_edit(
            "Patient presents with fever and cough",
            "en"
        )
        
        assert 'is_valid' in result
        assert 'medical_accuracy' in result
        assert 'safety_score' in result
    
    def test_scientific_validation(self):
        """Test scientific domain validation"""
        factory = DomainModuleFactory()
        modules = factory.create_default_modules()
        
        scientific_module = modules['scientific']
        result = scientific_module.validate_edit(
            "We hypothesize that E = mc²",
            "en"
        )
        
        assert 'is_valid' in result
        assert 'scientific_accuracy' in result


class TestLevel3Integration:
    """Test Level 3 integrated agent"""
    
    def test_agent_initialization(self):
        """Test Level 3 agent initialization"""
        config = Level3Config()
        agent = Level3Agent(config)
        
        assert agent.quantum_transformer is not None
        assert agent.rl_engine is not None
        assert agent.multimodal_processor is not None
        assert agent.backend_optimizer is not None
        assert agent.memory_store is not None
        assert len(agent.domain_modules) == 3
    
    def test_process_edit(self):
        """Test edit processing"""
        config = Level3Config()
        agent = Level3Agent(config)
        
        result = agent.process_edit(
            edit_content="Test edit content",
            language="english",
            rank=128,
            domain="legal",
            contributor_id="test_contributor"
        )
        
        assert 'overall_score' in result
        assert 'stages' in result
        assert 0 <= result['overall_score'] <= 1
    
    def test_benchmark(self):
        """Test benchmark execution"""
        config = Level3Config()
        agent = Level3Agent(config)
        
        test_cases = [
            {
                'edit_content': 'Test edit 1',
                'language': 'english',
                'rank': 128,
                'domain': 'legal'
            },
            {
                'edit_content': 'Test edit 2',
                'language': 'chinese',
                'rank': 64,
                'domain': 'medical'
            }
        ]
        
        results = agent.run_level3_benchmark(test_cases)
        
        assert 'total_cases' in results
        assert results['total_cases'] == 2
        assert 'avg_overall_score' in results
    
    def test_system_status(self):
        """Test system status"""
        config = Level3Config()
        agent = Level3Agent(config)
        
        status = agent.get_system_status()
        
        assert status['level'] == 3
        assert 'stages_enabled' in status
        assert sum(status['stages_enabled'].values()) >= 6


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
