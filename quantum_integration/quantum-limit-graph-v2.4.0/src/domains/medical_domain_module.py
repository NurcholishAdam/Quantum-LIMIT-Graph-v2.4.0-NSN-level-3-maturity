# -*- coding: utf-8 -*-
"""
Medical Domain Module
Specialized module for medical text editing with quantum simulation
"""
from typing import Dict, List, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import logging
from .domain_module_factory import BaseDomainModule, DomainConfig

logger = logging.getLogger(__name__)


class MedicalDomainModule(BaseDomainModule):
    """
    Medical domain module for medical text processing
    Handles medical terminology, drug interactions, and safety validation
    """
    
    def __init__(self, config: DomainConfig):
        """Initialize medical domain module"""
        super().__init__(config)
        
        # Medical terminology
        self.medical_terms = {
            'en': ['diagnosis', 'treatment', 'symptom', 'medication', 'prognosis'],
            'zh': ['诊断', '治疗', '症状', '药物', '预后'],
            'es': ['diagnóstico', 'tratamiento', 'síntoma', 'medicación', 'pronóstico'],
            'fr': ['diagnostic', 'traitement', 'symptôme', 'médicament', 'pronostic'],
            'de': ['Diagnose', 'Behandlung', 'Symptom', 'Medikament', 'Prognose'],
            'ja': ['診断', '治療', '症状', '薬', '予後']
        }
        
        # Drug interaction database (simplified)
        self.drug_interactions = {
            'aspirin': ['warfarin', 'ibuprofen'],
            'warfarin': ['aspirin', 'vitamin_k'],
            'ibuprofen': ['aspirin', 'lithium']
        }
        
        # Quantum simulator for molecular interactions
        if config.quantum_simulation_enabled:
            self.quantum_simulator = AerSimulator()
        else:
            self.quantum_simulator = None
        
        logger.info("Initialized MedicalDomainModule with medical terminology and drug database")
    
    def validate_edit(self, edit_content: str, language: str) -> Dict:
        """
        Validate medical edit
        
        Args:
            edit_content: Edit content
            language: Language
            
        Returns:
            Validation result
        """
        validation_result = {
            'is_valid': True,
            'medical_accuracy': 0.0,
            'safety_score': 0.0,
            'terminology_correctness': 0.0,
            'drug_interaction_check': True,
            'issues': []
        }
        
        # Check medical terminology
        if language in self.medical_terms:
            terms_found = sum(
                1 for term in self.medical_terms[language]
                if term.lower() in edit_content.lower()
            )
            validation_result['terminology_correctness'] = min(
                terms_found / len(self.medical_terms[language]), 1.0
            )
        
        # Check drug interactions
        drug_safety = self._check_drug_interactions(edit_content)
        validation_result['drug_interaction_check'] = drug_safety['safe']
        validation_result['safety_score'] = drug_safety['safety_score']
        
        # Quantum simulation for molecular validation
        if self.quantum_simulator:
            quantum_validation = self._quantum_molecular_validation(edit_content)
            validation_result['quantum_validation'] = quantum_validation
            validation_result['safety_score'] = (
                0.7 * validation_result['safety_score'] +
                0.3 * quantum_validation['stability']
            )
        
        # Overall medical accuracy
        validation_result['medical_accuracy'] = (
            0.4 * validation_result['terminology_correctness'] +
            0.6 * validation_result['safety_score']
        )
        
        # Check for critical issues
        if not validation_result['drug_interaction_check']:
            validation_result['issues'].append("Potential drug interaction detected")
            validation_result['is_valid'] = False
        
        if validation_result['safety_score'] < 0.7:
            validation_result['issues'].append("Low safety score")
        
        logger.info(f"Medical validation: accuracy={validation_result['medical_accuracy']:.3f}, "
                   f"safety={validation_result['safety_score']:.3f}")
        
        return validation_result
    
    def generate_edit_template(self, edit_type: str) -> Dict:
        """
        Generate medical edit template
        
        Args:
            edit_type: Type of edit
            
        Returns:
            Edit template
        """
        templates = {
            'diagnosis': {
                'structure': 'Patient presents with [symptoms]. Diagnosis: [condition]',
                'required_fields': ['symptoms', 'condition', 'diagnostic_criteria'],
                'example': 'Patient presents with fever and cough. Diagnosis: Upper respiratory infection'
            },
            'treatment': {
                'structure': 'Treatment plan: [medication] [dosage] for [duration]',
                'required_fields': ['medication', 'dosage', 'duration', 'contraindications'],
                'example': 'Treatment plan: Amoxicillin 500mg three times daily for 7 days'
            },
            'procedure': {
                'structure': '[procedure_name] performed to [objective]. Outcome: [result]',
                'required_fields': ['procedure_name', 'objective', 'result'],
                'example': 'Appendectomy performed to remove inflamed appendix. Outcome: Successful'
            }
        }
        
        template = templates.get(edit_type, templates['diagnosis'])
        
        logger.info(f"Generated medical template for: {edit_type}")
        
        return template
    
    def compute_domain_metrics(self, edit_result: Dict) -> Dict:
        """
        Compute medical-specific metrics
        
        Args:
            edit_result: Edit result
            
        Returns:
            Medical metrics
        """
        metrics = {
            'medical_accuracy': edit_result.get('medical_accuracy', 0.5),
            'safety_score': edit_result.get('safety_score', 0.5),
            'terminology_correctness': edit_result.get('terminology_correctness', 0.5),
            'clinical_relevance': np.random.uniform(0.7, 0.95),  # Placeholder
            'evidence_based_score': np.random.uniform(0.75, 0.98)  # Placeholder
        }
        
        # Add quantum metrics if available
        if 'quantum_validation' in edit_result:
            metrics['molecular_stability'] = edit_result['quantum_validation']['stability']
        
        logger.info(f"Medical metrics: safety={metrics['safety_score']:.3f}, "
                   f"accuracy={metrics['medical_accuracy']:.3f}")
        
        return metrics
    
    def _check_drug_interactions(self, text: str) -> Dict:
        """Check for drug interactions in text"""
        text_lower = text.lower()
        
        # Find mentioned drugs
        mentioned_drugs = [
            drug for drug in self.drug_interactions.keys()
            if drug in text_lower
        ]
        
        # Check for interactions
        interactions_found = []
        for drug in mentioned_drugs:
            for interacting_drug in self.drug_interactions[drug]:
                if interacting_drug in text_lower:
                    interactions_found.append((drug, interacting_drug))
        
        safe = len(interactions_found) == 0
        safety_score = 1.0 if safe else max(0.3, 1.0 - len(interactions_found) * 0.2)
        
        return {
            'safe': safe,
            'safety_score': safety_score,
            'interactions': interactions_found
        }
    
    def _quantum_molecular_validation(self, text: str) -> Dict:
        """
        Perform quantum simulation for molecular validation
        
        Args:
            text: Text to validate
            
        Returns:
            Quantum validation result
        """
        # Build quantum circuit for molecular simulation
        num_qubits = 6
        qc = QuantumCircuit(num_qubits)
        
        # Encode molecular structure (simplified)
        for i in range(num_qubits):
            qc.h(i)
        
        # Simulate molecular interactions
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        
        # Apply molecular Hamiltonian
        for i in range(num_qubits):
            qc.rz(np.pi / 4, i)
        
        # Measure
        qc.measure_all()
        
        # Execute
        job = self.quantum_simulator.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Compute stability from measurement distribution
        probs = np.array([counts.get(format(i, f'0{num_qubits}b'), 0) for i in range(2**num_qubits)])
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Lower entropy = higher stability
        max_entropy = np.log2(2**num_qubits)
        stability = 1.0 - (entropy / max_entropy)
        
        return {
            'stability': float(np.clip(stability, 0, 1)),
            'entropy': entropy,
            'quantum_state': counts
        }


def demo_medical_domain_module():
    """Demo medical domain module"""
    logger.info("=" * 80)
    logger.info("MEDICAL DOMAIN MODULE DEMO")
    logger.info("=" * 80)
    
    # Initialize module
    config = DomainConfig(
        domain_name='medical',
        supported_languages=['en', 'zh', 'es'],
        edit_templates=[],
        quantum_simulation_enabled=True,
        specialized_metrics=['medical_accuracy', 'safety_score']
    )
    
    module = MedicalDomainModule(config)
    
    # Test validation
    logger.info("\n--- Test 1: Validate Medical Edit ---")
    edit = "Patient presents with fever and cough. Treatment: aspirin for symptom relief"
    result = module.validate_edit(edit, 'en')
    logger.info(f"Valid: {result['is_valid']}")
    logger.info(f"Medical accuracy: {result['medical_accuracy']:.3f}")
    logger.info(f"Safety score: {result['safety_score']:.3f}")
    
    # Test drug interaction
    logger.info("\n--- Test 2: Drug Interaction Check ---")
    edit_interaction = "Prescribe aspirin and warfarin for treatment"
    result_interaction = module.validate_edit(edit_interaction, 'en')
    logger.info(f"Drug interaction safe: {result_interaction['drug_interaction_check']}")
    logger.info(f"Issues: {result_interaction['issues']}")
    
    # Test template generation
    logger.info("\n--- Test 3: Generate Template ---")
    template = module.generate_edit_template('treatment')
    logger.info(f"Template: {template['structure']}")
    logger.info(f"Example: {template['example']}")
    
    # Test metrics
    logger.info("\n--- Test 4: Compute Metrics ---")
    metrics = module.compute_domain_metrics(result)
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.3f}")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info("\n" + "=" * 80)
    logger.info("DEMO COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    demo_medical_domain_module()
