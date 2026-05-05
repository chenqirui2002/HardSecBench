"""
Agent implementations

Architecture:
- base_agent: Abstract base classes for all agents
- generation/: Code and test generation agents (Expert, RedTeamer, Architect)
- evaluation/: Analysis and evaluation agents (Arbiter)
- orchestration/: Workflow coordination agents (IterativeGenerator)
"""
# Base classes
from .base_agent import (
    BaseAgent,
    GenerationAgent,
    EvaluationAgent,
    OrchestrationAgent
)

# New architecture exports
from .generation import ArchitectAgent, ExpertAgent, RedTeamerAgent
from .evaluation import ArbiterAgent
from .orchestration import IterativeCodeGenerator

__all__ = [
    # Base classes
    'BaseAgent',
    'GenerationAgent',
    'EvaluationAgent',
    'OrchestrationAgent',
    # Generation agents
    'ArchitectAgent',
    'ExpertAgent',
    'RedTeamerAgent',
    # Evaluation agents
    'ArbiterAgent',
    # Orchestration
    'IterativeCodeGenerator',
]
