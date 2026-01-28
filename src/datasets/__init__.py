"""
Dataset loading utilities.

Provides functions to download and prepare UCI ML Repository datasets.
"""

from .loaders import (
    download_mushroom_dataset,
    download_tic_tac_toe_dataset,
    download_voting_dataset,
    train_test_split,
    evaluate_model,
    test_dataset
)

__all__ = [
    'download_mushroom_dataset',
    'download_tic_tac_toe_dataset',
    'download_voting_dataset',
    'train_test_split',
    'evaluate_model',
    'test_dataset'
]
