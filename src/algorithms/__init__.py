"""
Core decision tree algorithm implementations.

Available algorithms:
- ID3 (1986): Information gain criterion
- C4.5 (1993): Gain ratio + pruning
"""

from .id3 import ID3
from .c45 import C45

__all__ = ['ID3', 'C45']
