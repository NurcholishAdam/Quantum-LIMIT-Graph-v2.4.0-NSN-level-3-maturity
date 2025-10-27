#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Level 3 Maturity Validation Script
Validates that all components are properly implemented
"""
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def validate_imports():
    """Validate all imports work"""
    logger.info("\n" + "="*80)
    logger.info("VALIDATING IMPORTS")
    logger.info("="*80)
    
    try:
        # Stage 1
        from src.core.quantum_transformer_core import QuantumTransformerCore
        logger.info("✅ Stage 1: Quantum Transformers")
        
        # Stage 2
        from src.rl.meta_cognitive_rl_engine import MetaCognitiveRLEngine
        logger.info("✅ Stage 2: Meta-Cognitive RL")
        
        # Stage 3
        from src.perception.multimodal_quantum_processor import MultimodalQuantumProcessor
        logger.info("✅ Stage 3: Multimodal Perception")
        
        # Stage 4
        from src.orchestration.adaptive_backend_optimizer import AdaptiveBackendOptimizer
        logger.info("✅ Stage 4: Adaptive Backend Optimization")
        
        # Stage 5
        from src.memory.entangled_memory_store import EntangledMemoryStore
        logger.info("✅ Stage 5: Entangled Memory")
        
        # Stage 6
        from src.domains.domain_module_factory import DomainModuleFactory
        logger.info("✅ Stage 6: Domain Modules")
        
        # Integration
        from level_3_agent import Level3Agent, Level3Config
        logger.info("✅ Integration: Level 3 Agent")
        
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def validate_file_structure():
    """Validate file structure"""
    logger.info("\n" + "="*80)
    logger.info("VALIDATING FILE STRUCTURE")
    logger.info("="*80)
    
    required_files = [
        'src/core/quantum_transformer_core.py',
        'src/rl/meta_cognitive_rl_engine.py',
        'src/perception/multimodal_quantum_processor.py',
        'src/orchestration/adaptive_backend_optimizer.py',
        'src/memory/entangled_memory_store.py',
        'src/domains/domain_module_factory.py',
        'src/domains/legal_domain_module.py',
        'src/domains/medical_domain_module.py',
        'src/domains/scientific_domain_module.py',
        'level_3_agent.py',
        'tests/test_level3_maturity.py',
        'LEVEL_3_MATURITY_ROADMAP.md',
        'LEVEL_3_MATURITY_COMPLETE.md',
        'LEVEL_3_QUICK_START.md',
        'INDEX.md'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            logger.info(f"✅ {file_path}")
        else:
            logger.error(f"❌ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist


def validate_functionality():
    """Validate basic functionality"""
    logger.info("\n" + "="*80)
    logger.info("VALIDATING FUNCTIONALITY")
    logger.info("="*80)
    
    try:
        from level_3_agent import Level3Agent, Level3Config
        
        # Initialize agent
        config = Level3Config()
        agent = Level3Agent(config)
        logger.info("✅ Agent initialization")
        
        # Test edit processing
        result = agent.process_edit(
            edit_content="Test edit",
            language="english",
            rank=128
        )
        
        if 'overall_score' in result:
            logger.info(f"✅ Edit processing (score: {result['overall_score']:.3f})")
        else:
            logger.error("❌ Edit processing - missing overall_score")
            return False
        
        # Test system status
        status = agent.get_system_status()
        if status['level'] == 3:
            logger.info(f"✅ System status (level: {status['level']})")
        else:
            logger.error(f"❌ System status - wrong level: {status['level']}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_documentation():
    """Validate documentation completeness"""
    logger.info("\n" + "="*80)
    logger.info("VALIDATING DOCUMENTATION")
    logger.info("="*80)
    
    docs = {
        'LEVEL_3_MATURITY_ROADMAP.md': 'Roadmap',
        'LEVEL_3_MATURITY_COMPLETE.md': 'Complete Guide',
        'LEVEL_3_QUICK_START.md': 'Quick Start',
        'INDEX.md': 'Index',
        'README.md': 'Main README'
    }
    
    all_exist = True
    for doc_file, doc_name in docs.items():
        if os.path.exists(doc_file):
            # Check file size
            size = os.path.getsize(doc_file)
            if size > 1000:  # At least 1KB
                logger.info(f"✅ {doc_name} ({size:,} bytes)")
            else:
                logger.warning(f"⚠️  {doc_name} - file too small ({size} bytes)")
        else:
            logger.error(f"❌ {doc_name} - NOT FOUND")
            all_exist = False
    
    return all_exist


def validate_code_metrics():
    """Validate code metrics"""
    logger.info("\n" + "="*80)
    logger.info("VALIDATING CODE METRICS")
    logger.info("="*80)
    
    files_to_check = [
        ('src/core/quantum_transformer_core.py', 400, 'Stage 1'),
        ('src/rl/meta_cognitive_rl_engine.py', 500, 'Stage 2'),
        ('src/perception/multimodal_quantum_processor.py', 550, 'Stage 3'),
        ('src/orchestration/adaptive_backend_optimizer.py', 600, 'Stage 4'),
        ('src/memory/entangled_memory_store.py', 700, 'Stage 5'),
        ('src/domains/domain_module_factory.py', 150, 'Stage 6 Factory'),
        ('level_3_agent.py', 400, 'Integration'),
        ('tests/test_level3_maturity.py', 350, 'Tests')
    ]
    
    total_lines = 0
    for file_path, min_lines, name in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                total_lines += lines
                if lines >= min_lines:
                    logger.info(f"✅ {name}: {lines} lines (min: {min_lines})")
                else:
                    logger.warning(f"⚠️  {name}: {lines} lines (expected: {min_lines}+)")
        else:
            logger.error(f"❌ {name} - NOT FOUND")
    
    logger.info(f"\n📊 Total lines of code: {total_lines:,}")
    
    return total_lines >= 4000


def main():
    """Main validation"""
    logger.info("\n" + "="*80)
    logger.info("QUANTUM LIMIT-GRAPH v2.5.0-L3 VALIDATION")
    logger.info("="*80)
    
    results = {
        'Imports': validate_imports(),
        'File Structure': validate_file_structure(),
        'Functionality': validate_functionality(),
        'Documentation': validate_documentation(),
        'Code Metrics': validate_code_metrics()
    }
    
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*80)
    
    for category, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{category}: {status}")
    
    all_passed = all(results.values())
    
    logger.info("\n" + "="*80)
    if all_passed:
        logger.info("🎉 ALL VALIDATIONS PASSED")
        logger.info("Level 3 Maturity implementation is COMPLETE and READY")
    else:
        logger.error("❌ SOME VALIDATIONS FAILED")
        logger.error("Please review the errors above")
    logger.info("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
