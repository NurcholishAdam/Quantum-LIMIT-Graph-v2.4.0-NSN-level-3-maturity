# -*- coding: utf-8 -*-
"""
Domain Module Factory
Creates domain-specific modules for specialized benchmarking
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DomainConfig:
    """Configuration for domain module"""
    domain_name: str
    supported_languages: List[str]
    edit_templates: List[Dict]
    quantum_simulation_enabled: bool
    specialized_metrics: List[str]


class BaseDomainModule:
    """Base class for domain modules"""
    
    def __init__(self, config: DomainConfig):
        """
        Initialize domain module
        
        Args:
            config: Domain configuration
        """
        self.config = config
        self.domain_knowledge = {}
        
        logger.info(f"Initialized {config.domain_name} domain module")
    
    def validate_edit(self, edit_content: str, language: str) -> Dict:
        """
        Validate edit for domain
        
        Args:
            edit_content: Edit content
            language: Language
            
        Returns:
            Validation result
        """
        raise NotImplementedError("Subclasses must implement validate_edit")
    
    def generate_edit_template(self, edit_type: str) -> Dict:
        """
        Generate domain-specific edit template
        
        Args:
            edit_type: Type of edit
            
        Returns:
            Edit template
        """
        raise NotImplementedError("Subclasses must implement generate_edit_template")
    
    def compute_domain_metrics(self, edit_result: Dict) -> Dict:
        """
        Compute domain-specific metrics
        
        Args:
            edit_result: Edit result
            
        Returns:
            Domain metrics
        """
        raise NotImplementedError("Subclasses must implement compute_domain_metrics")


class DomainModuleFactory:
    """
    Factory for creating domain-specific modules
    """
    
    def __init__(self):
        """Initialize domain module factory"""
        self.registered_domains: Dict[str, type] = {}
        self.active_modules: Dict[str, BaseDomainModule] = {}
        
        logger.info("Initialized DomainModuleFactory")
    
    def register_domain(self, domain_name: str, module_class: type):
        """
        Register a domain module
        
        Args:
            domain_name: Domain name
            module_class: Module class
        """
        self.registered_domains[domain_name] = module_class
        logger.info(f"Registered domain: {domain_name}")
    
    def create_module(self, domain_name: str, config: DomainConfig) -> BaseDomainModule:
        """
        Create domain module
        
        Args:
            domain_name: Domain name
            config: Domain configuration
            
        Returns:
            Domain module instance
        """
        if domain_name not in self.registered_domains:
            raise ValueError(f"Domain {domain_name} not registered")
        
        module_class = self.registered_domains[domain_name]
        module = module_class(config)
        
        self.active_modules[domain_name] = module
        
        logger.info(f"Created module for domain: {domain_name}")
        
        return module
    
    def get_module(self, domain_name: str) -> Optional[BaseDomainModule]:
        """Get active module for domain"""
        return self.active_modules.get(domain_name)
    
    def list_available_domains(self) -> List[str]:
        """List all available domains"""
        return list(self.registered_domains.keys())
    
    def create_default_modules(self) -> Dict[str, BaseDomainModule]:
        """Create default domain modules"""
        from .legal_domain_module import LegalDomainModule
        from .medical_domain_module import MedicalDomainModule
        from .scientific_domain_module import ScientificDomainModule
        
        # Register default domains
        self.register_domain('legal', LegalDomainModule)
        self.register_domain('medical', MedicalDomainModule)
        self.register_domain('scientific', ScientificDomainModule)
        
        # Create modules with default configs
        modules = {}
        
        # Legal
        legal_config = DomainConfig(
            domain_name='legal',
            supported_languages=['en', 'zh', 'es', 'fr', 'de'],
            edit_templates=[],
            quantum_simulation_enabled=False,
            specialized_metrics=['legal_accuracy', 'citation_validity']
        )
        modules['legal'] = self.create_module('legal', legal_config)
        
        # Medical
        medical_config = DomainConfig(
            domain_name='medical',
            supported_languages=['en', 'zh', 'es', 'fr', 'de', 'ja'],
            edit_templates=[],
            quantum_simulation_enabled=True,
            specialized_metrics=['medical_accuracy', 'safety_score', 'terminology_correctness']
        )
        modules['medical'] = self.create_module('medical', medical_config)
        
        # Scientific
        scientific_config = DomainConfig(
            domain_name='scientific',
            supported_languages=['en', 'zh', 'ru', 'de', 'fr'],
            edit_templates=[],
            quantum_simulation_enabled=True,
            specialized_metrics=['scientific_accuracy', 'formula_correctness', 'unit_consistency']
        )
        modules['scientific'] = self.create_module('scientific', scientific_config)
        
        logger.info(f"Created {len(modules)} default domain modules")
        
        return modules


def demo_domain_module_factory():
    """Demo domain module factory"""
    logger.info("=" * 80)
    logger.info("DOMAIN MODULE FACTORY DEMO")
    logger.info("=" * 80)
    
    # Initialize factory
    factory = DomainModuleFactory()
    
    # Create default modules
    logger.info("\n--- Creating Default Modules ---")
    modules = factory.create_default_modules()
    
    for domain_name, module in modules.items():
        logger.info(f"Created: {domain_name} - {module.config.domain_name}")
    
    # List available domains
    logger.info("\n--- Available Domains ---")
    domains = factory.list_available_domains()
    logger.info(f"Domains: {domains}")
    
    logger.info("\n" + "=" * 80)
    logger.info("DEMO COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    demo_domain_module_factory()
