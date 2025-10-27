# -*- coding: utf-8 -*-
"""
Stage 6: Domain-Aware Modules
Domain-specific quantum modules for specialized benchmarking
"""
from .domain_module_factory import DomainModuleFactory
from .legal_domain_module import LegalDomainModule
from .medical_domain_module import MedicalDomainModule
from .scientific_domain_module import ScientificDomainModule

__all__ = [
    'DomainModuleFactory',
    'LegalDomainModule',
    'MedicalDomainModule',
    'ScientificDomainModule'
]
