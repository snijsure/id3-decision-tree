"""
Test ID3 algorithm on classic UCI Machine Learning Repository datasets.
All datasets selected have categorical/discrete attributes suitable for ID3.
"""

import urllib.request
import csv
from typing import List, Dict, Any, Tuple
import random


def download_mushroom_dataset() -> List[Dict[str, Any]]:
    """
    Download and parse the Mushroom dataset from UCI ML Repository.

    Dataset: https://archive.ics.uci.edu/ml/datasets/Mushroom
    - 8124 instances
    - 22 categorical attributes
    - Binary classification: edible (e) vs poisonous (p)

    Returns:
        List of example dictionaries
    """
    print("\nDownloading Mushroom dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

    # Attribute names from the dataset description
    attributes = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat'
    ]

    examples = []
    try:
        with urllib.request.urlopen(url) as response:
            lines = response.read().decode('utf-8').strip().split('\n')

        for line in lines:
            if not line.strip():
                continue
            values = line.strip().split(',')
            if len(values) == len(attributes):
                example = dict(zip(attributes, values))
                examples.append(example)

        print(f"  Loaded {len(examples)} examples with {len(attributes)-1} attributes")
        return examples
    except Exception as e:
        print(f"  Error downloading: {e}")
        return []


def download_tic_tac_toe_dataset() -> List[Dict[str, Any]]:
    """
    Download and parse the Tic-Tac-Toe Endgame dataset from UCI ML Repository.

    Dataset: https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
    - 958 instances
    - 9 categorical attributes (board positions)
    - Binary classification: positive (x wins) vs negative

    Returns:
        List of example dictionaries
    """
    print("\nDownloading Tic-Tac-Toe Endgame dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data"

    # Positions on the tic-tac-toe board
    attributes = ['top-left', 'top-middle', 'top-right',
                  'middle-left', 'middle-middle', 'middle-right',
                  'bottom-left', 'bottom-middle', 'bottom-right',
                  'class']

    examples = []
    try:
        with urllib.request.urlopen(url) as response:
            lines = response.read().decode('utf-8').strip().split('\n')

        for line in lines:
            if not line.strip():
                continue
            values = line.strip().split(',')
            if len(values) == len(attributes):
                example = dict(zip(attributes, values))
                examples.append(example)

        print(f"  Loaded {len(examples)} examples with {len(attributes)-1} attributes")
        return examples
    except Exception as e:
        print(f"  Error downloading: {e}")
        return []


def download_voting_dataset() -> List[Dict[str, Any]]:
    """
    Download and parse the Congressional Voting Records dataset.

    Dataset: https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
    - 435 instances
    - 16 binary attributes (votes on 16 issues)
    - Binary classification: democrat vs republican

    Returns:
        List of example dictionaries
    """
    print("\nDownloading Congressional Voting Records dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"

    attributes = [
        'class', 'handicapped-infants', 'water-project-cost-sharing',
        'adoption-of-the-budget-resolution', 'physician-fee-freeze',
        'el-salvador-aid', 'religious-groups-in-schools',
        'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
        'mx-missile', 'immigration', 'synfuels-corporation-cutback',
        'education-spending', 'superfund-right-to-sue', 'crime',
        'duty-free-exports', 'export-administration-act-south-africa'
    ]

    examples = []
    try:
        with urllib.request.urlopen(url) as response:
            lines = response.read().decode('utf-8').strip().split('\n')

        for line in lines:
            if not line.strip():
                continue
            values = line.strip().split(',')
            if len(values) == len(attributes):
                example = dict(zip(attributes, values))
                examples.append(example)

        print(f"  Loaded {len(examples)} examples with {len(attributes)-1} attributes")
        return examples
    except Exception as e:
        print(f"  Error downloading: {e}")
        return []


def train_test_split(data: List[Dict[str, Any]],
                     test_ratio: float = 0.3) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into training and test sets.

    Args:
        data: Full dataset
        test_ratio: Proportion to use for testing

    Returns:
        (training_set, test_set)
    """
    shuffled = data.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def evaluate_model(model: Any, test_set: List[Dict[str, Any]], class_attr: str = 'class') -> float:
    """
    Evaluate model accuracy on test set.

    Args:
        model: Trained model (ID3, C4.5, or any model with predict method)
        test_set: Test examples
        class_attr: Name of class attribute

    Returns:
        Accuracy (0.0 to 1.0)
    """
    correct = 0
    total = 0

    for example in test_set:
        test_example = {k: v for k, v in example.items() if k != class_attr}
        prediction = model.predict(test_example)

        if prediction == example[class_attr]:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def test_dataset(name: str, examples: List[Dict[str, Any]], class_attr: str = 'class'):
    """
    Test ID3 on a dataset using train/test split.

    Args:
        name: Dataset name
        examples: All examples
        class_attr: Name of class attribute
    """
    if not examples:
        print(f"\n{name}: Skipping (no data)")
        return

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    # Split data
    train_set, test_set = train_test_split(examples, test_ratio=0.3)
    print(f"Training set: {len(train_set)} examples")
    print(f"Test set:     {len(test_set)} examples")

    # Count classes
    class_counts = {}
    for ex in examples:
        cls = ex[class_attr]
        class_counts[cls] = class_counts.get(cls, 0) + 1
    print(f"Classes: {class_counts}")

    # Train model
    print("\nTraining ID3...")
    model = ID3()
    try:
        model.fit(train_set, class_attr=class_attr)

        # Evaluate
        train_accuracy = evaluate_model(model, train_set, class_attr)
        test_accuracy = evaluate_model(model, test_set, class_attr)

        print(f"\nResults:")
        print(f"  Training accuracy: {train_accuracy*100:.2f}%")
        print(f"  Test accuracy:     {test_accuracy*100:.2f}%")

        # Count tree size
        def count_nodes(node):
            if node.is_leaf():
                return 1
            return 1 + sum(count_nodes(child) for child in node.children.values())

        tree_size = count_nodes(model.root)
        print(f"  Tree size:         {tree_size} nodes")

    except Exception as e:
        print(f"  Error training: {e}")


def main():
    """Run tests on multiple datasets."""
    print("="*60)
    print("Testing ID3 on UCI Machine Learning Repository Datasets")
    print("="*60)

    # Set random seed for reproducibility
    random.seed(42)

    # Test Mushroom dataset
    mushroom_data = download_mushroom_dataset()
    if mushroom_data:
        test_dataset("Mushroom Classification", mushroom_data, class_attr='class')

    # Test Tic-Tac-Toe dataset
    tictactoe_data = download_tic_tac_toe_dataset()
    if tictactoe_data:
        test_dataset("Tic-Tac-Toe Endgame", tictactoe_data, class_attr='class')

    # Test Voting dataset
    voting_data = download_voting_dataset()
    if voting_data:
        test_dataset("Congressional Voting Records", voting_data, class_attr='class')

    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)


if __name__ == "__main__":
    main()
