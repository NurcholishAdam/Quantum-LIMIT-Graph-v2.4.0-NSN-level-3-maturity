# -*- coding: utf-8 -*-
"""
Legal Domain Module
Specialized module for legal text editing and validation
"""
from typing import Dict, List, Optional
import numpy as np
import logging
from .domain_module_factory import BaseDomainModule, DomainConfig

logger = logging.getLogger(__name__)


class LegalDomainModule(BaseDomainModule):
    """
    Legal domain module for legal text processing
    Handles legal terminology, citations, and compliance
    """
    
    def __init__(self, config: DomainConfig):
        """Initialize legal domain module"""
        super().__init__(config)
        
        # Legal-specific knowledge
        self.legal_terms = {
            'en': ['plaintiff', 'defendant', 'jurisdiction', 'statute', 'precedent'],
            'zh': ['原告', '被告', '管辖权', '法规', '先例'],
            'es': ['demandante', 'demandado', 'jurisdicción', 'estatuto', 'precedente'],
            'fr': ['demandeur', 'défendeur', 'juridiction', 'statut', 'précédent'],
            'de': ['Kläger', 'Beklagter', 'Zuständigkeit', 'Gesetz', 'Präzedenzfall']
        }
        
        self.citation_patterns = {
            'en': r'\d+\s+[A-Z][a-z]+\s+\d+',  # e.g., "123 Smith 456"
            'zh': r'[\u4e00-\u9fff]+法\[\d+\]号',  # Chinese legal citation
        }
        
        logger.info("Initialized LegalDomainModule with legal terminology database")
    
    def validate_edit(self, edit_content: str, language: str) -> Dict:
        """
        Validate legal edit
        
        Args:
            edit_content: Edit content
            language: Language
            
        Returns:
            Validation result
        """
        validation_result = {
            'is_valid': True,
            'legal_accuracy': 0.0,
            'citation_validity': 0.0,
            'terminology_correctness': 0.0,
            'issues': []
        }
        
        # Check legal terminology
        if language in self.legal_terms:
            terms_found = sum(
                1 for term in self.legal_terms[language]
                if term.lower() in edit_content.lower()
            )
            validation_result['terminology_correctness'] = min(
                terms_found / len(self.legal_terms[language]), 1.0
            )
        
        # Check citation format
        citation_score = self._validate_citations(edit_content, language)
        validation_result['citation_validity'] = citation_score
        
        # Overall legal accuracy
        validation_result['legal_accuracy'] = (
            0.6 * validation_result['terminology_correctness'] +
            0.4 * validation_result['citation_validity']
        )
        
        # Check for issues
        if validation_result['legal_accuracy'] < 0.5:
            validation_result['issues'].append("Low legal accuracy")
            validation_result['is_valid'] = False
        
        logger.info(f"Legal validation: accuracy={validation_result['legal_accuracy']:.3f}")
        
        return validation_result
    
    def generate_edit_template(self, edit_type: str) -> Dict:
        """
        Generate legal edit template
        
        Args:
            edit_type: Type of edit
            
        Returns:
            Edit template
        """
        templates = {
            'case_law': {
                'structure': 'In [case_name], the court held that [ruling]',
                'required_fields': ['case_name', 'ruling', 'citation'],
                'example': 'In Smith v. Jones, the court held that contracts require consideration'
            },
            'statute': {
                'structure': 'According to [statute_name] § [section], [provision]',
                'required_fields': ['statute_name', 'section', 'provision'],
                'example': 'According to the Civil Code § 1234, parties must act in good faith'
            },
            'definition': {
                'structure': '[term] means [definition] under [jurisdiction] law',
                'required_fields': ['term', 'definition', 'jurisdiction'],
                'example': 'Negligence means failure to exercise reasonable care under tort law'
            }
        }
        
        template = templates.get(edit_type, templates['definition'])
        
        logger.info(f"Generated legal template for: {edit_type}")
        
        return template
    
    def compute_domain_metrics(self, edit_result: Dict) -> Dict:
        """
        Compute legal-specific metrics
        
        Args:
            edit_result: Edit result
            
        Returns:
            Legal metrics
        """
        metrics = {
            'legal_accuracy': edit_result.get('legal_accuracy', 0.5),
            'citation_validity': edit_result.get('citation_validity', 0.5),
            'terminology_correctness': edit_result.get('terminology_correctness', 0.5),
            'compliance_score': self._compute_compliance_score(edit_result),
            'precedent_alignment': np.random.uniform(0.7, 0.95)  # Placeholder
        }
        
        logger.info(f"Legal metrics: compliance={metrics['compliance_score']:.3f}")
        
        return metrics
    
    def _validate_citations(self, text: str, language: str) -> float:
        """Validate legal citations in text"""
        import re
        
        if language not in self.citation_patterns:
            return 0.5  # Default score for unsupported languages
        
        pattern = self.citation_patterns[language]
        citations = re.findall(pattern, text)
        
        if not citations:
            return 0.3  # Low score if no citations found
        
        # Score based on citation count and format
        score = min(len(citations) / 3.0, 1.0)
        
        return score
    
    def _compute_compliance_score(self, edit_result: Dict) -> float:
        """Compute legal compliance score"""
        accuracy = edit_result.get('legal_accuracy', 0.5)
        citation = edit_result.get('citation_validity', 0.5)
        terminology = edit_result.get('terminology_correctness', 0.5)
        
        compliance = 0.4 * accuracy + 0.3 * citation + 0.3 * terminology
        
        return float(np.clip(compliance, 0, 1))


def demo_legal_domain_module():
    """Demo legal domain module"""
    logger.info("=" * 80)
    logger.info("LEGAL DOMAIN MODULE DEMO")
    logger.info("=" * 80)
    
    # Initialize module
    config = DomainConfig(
        domain_name='legal',
        supported_languages=['en', 'zh', 'es'],
        edit_templates=[],
        quantum_simulation_enabled=False,
        specialized_metrics=['legal_accuracy', 'citation_validity']
    )
    
    module = LegalDomainModule(config)
    
    # Test validation
    logger.info("\n--- Test 1: Validate Legal Edit ---")
    edit = "In Smith v. Jones, the plaintiff argued that the defendant violated the statute"
    result = module.validate_edit(edit, 'en')
    logger.info(f"Valid: {result['is_valid']}")
    logger.info(f"Legal accuracy: {result['legal_accuracy']:.3f}")
    
    # Test template generation
    logger.info("\n--- Test 2: Generate Template ---")
    template = module.generate_edit_template('case_law')
    logger.info(f"Template: {template['structure']}")
    logger.info(f"Example: {template['example']}")
    
    # Test metrics
    logger.info("\n--- Test 3: Compute Metrics ---")
    metrics = module.compute_domain_metrics(result)
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.3f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("DEMO COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    demo_legal_domain_module()
