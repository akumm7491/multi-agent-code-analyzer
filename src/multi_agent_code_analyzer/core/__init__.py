"""
Core module for DDD code analysis.
This module provides functionality for analyzing code repositories
using Domain-Driven Design principles.
"""

from .domain_analysis import (
    DomainAnalyzer,
    identify_bounded_context,
    detect_ddd_patterns,
    calculate_metrics
)

__all__ = [
    'DomainAnalyzer',
    'identify_bounded_context',
    'detect_ddd_patterns',
    'calculate_metrics'
]
