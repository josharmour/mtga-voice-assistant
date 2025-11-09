"""
RL Algorithm Implementations

This module contains reinforcement learning algorithms optimized
for Magic: The Gathering gameplay scenarios.
"""

from .base import BaseRLAlgorithm, BaseQAlgorithm
from .cql import ConservativeQLearning, create_conservative_q_learning

__all__ = ['BaseRLAlgorithm', 'BaseQAlgorithm', 'ConservativeQLearning', 'create_conservative_q_learning']