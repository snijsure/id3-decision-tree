"""
Analysis of ID3 behavior on different training set sizes.
Demonstrates overfitting and the importance of proper evaluation.
"""

import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
from typing import List, Dict, Any
from src.algorithms import ID3
from src.datasets import download_mushroom_dataset, train_test_split, evaluate_model

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def count_tree_nodes(node) -> int:
    """Count total nodes in a tree."""
    if node is None or node.is_leaf():
        return 1
    return 1 + sum(count_tree_nodes(child) for child in node.children.values())


def analyze_training_size_effect(examples: List[Dict[str, Any]],
                                 class_attr: str = 'class'):
    """
    Analyze how training set size affects model performance.

    Demonstrates the overfitting phenomenon discussed in Quinlan's paper.
    """
    print("\n" + "="*70)
    print("Analysis: Effect of Training Set Size")
    print("="*70)
    print("\nThis demonstrates a key insight from Quinlan's paper:")
    print("ID3 achieves 100% training accuracy but may overfit small datasets.\n")

    # Use fixed test set
    random.seed(42)
    all_shuffled = examples.copy()
    random.shuffle(all_shuffled)

    # Reserve 30% for testing
    split_idx = int(len(all_shuffled) * 0.7)
    available_train = all_shuffled[:split_idx]
    test_set = all_shuffled[split_idx:]

    # Try different training set sizes
    train_sizes = [50, 100, 200, 500, 1000, 2000, len(available_train)]
    results = []

    print(f"{'Size':<8} {'Train Acc':<12} {'Test Acc':<12} {'Tree Size':<12} {'Overfitting':<12}")
    print("-" * 70)

    for size in train_sizes:
        if size > len(available_train):
            continue

        # Sample training set
        train_set = available_train[:size]

        # Train model
        model = ID3()
        model.fit(train_set, class_attr=class_attr)

        # Evaluate
        train_acc = evaluate_model(model, train_set, class_attr)
        test_acc = evaluate_model(model, test_set, class_attr)
        tree_size = count_tree_nodes(model.root)
        overfitting = train_acc - test_acc

        results.append({
            'size': size,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'tree_size': tree_size,
            'overfitting': overfitting
        })

        print(f"{size:<8} {train_acc*100:>10.2f}% {test_acc*100:>10.2f}% "
              f"{tree_size:>10} {overfitting*100:>10.2f}%")

    return results


def analyze_attribute_importance(examples: List[Dict[str, Any]],
                                 class_attr: str = 'class'):
    """
    Show information gain for each attribute.

    Demonstrates the information-theoretic attribute selection from Section 4.
    """
    print("\n" + "="*70)
    print("Analysis: Attribute Information Gain")
    print("="*70)
    print("\nShows which attributes are most informative (Section 4 of paper):")
    print("Attributes with higher information gain are selected first.\n")

    # Sample subset for faster computation
    sample_size = min(1000, len(examples))
    sample = random.sample(examples, sample_size)

    # Calculate information gain for each attribute
    model = ID3()
    attributes = [attr for attr in sample[0].keys() if attr != class_attr]

    gains = []
    for attr in attributes:
        gain = model._information_gain(sample, attr, class_attr)
        gains.append((attr, gain))

    # Sort by gain
    gains.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Rank':<6} {'Attribute':<30} {'Info Gain':<12}")
    print("-" * 70)

    for i, (attr, gain) in enumerate(gains[:15], 1):  # Top 15
        print(f"{i:<6} {attr:<30} {gain:>10.4f} bits")

    if len(gains) > 15:
        print(f"\n... and {len(gains) - 15} more attributes")

    return gains


def plot_results(results: List[Dict], output_file: str = 'analysis_plot.png'):
    """Create visualization of training size analysis."""
    if not HAS_MATPLOTLIB:
        print("\n⚠️  Matplotlib not available. Skipping plot generation.")
        print("    Install with: pip install matplotlib")
        return

    sizes = [r['size'] for r in results]
    train_accs = [r['train_acc'] * 100 for r in results]
    test_accs = [r['test_acc'] * 100 for r in results]
    tree_sizes = [r['tree_size'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy vs Training Size
    ax1.plot(sizes, train_accs, 'o-', label='Training Accuracy', linewidth=2, markersize=8)
    ax1.plot(sizes, test_accs, 's-', label='Test Accuracy', linewidth=2, markersize=8)
    ax1.set_xlabel('Training Set Size', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('ID3 Performance vs Training Set Size', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([70, 105])

    # Plot 2: Tree Size vs Training Size
    ax2.plot(sizes, tree_sizes, 'o-', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Training Set Size', fontsize=12)
    ax2.set_ylabel('Number of Nodes', fontsize=12)
    ax2.set_title('Decision Tree Complexity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_file}")


def main():
    """Run comprehensive analysis."""
    print("\n" + "="*70)
    print("ID3 Algorithm Analysis")
    print("Based on Quinlan's 1986 Paper: 'Induction of Decision Trees'")
    print("="*70)

    # Download dataset
    print("\nLoading Mushroom dataset for analysis...")
    examples = download_mushroom_dataset()

    if not examples:
        print("Failed to load dataset")
        return

    print(f"Dataset loaded: {len(examples)} examples")

    # Analysis 1: Training size effect
    results = analyze_training_size_effect(examples, class_attr='class')

    # Analysis 2: Attribute importance
    random.seed(42)
    gains = analyze_attribute_importance(examples, class_attr='class')

    # Generate plot
    plot_results(results)

    # Summary
    print("\n" + "="*70)
    print("Key Insights from Quinlan's Paper")
    print("="*70)
    print("""
1. PERFECT TRAINING ACCURACY: ID3 always achieves 100% on training data
   (assuming adequate attributes). This is both a strength and weakness.

2. OVERFITTING: Small training sets → complex trees → poor generalization.
   The gap between training and test accuracy indicates overfitting.

3. INFORMATION GAIN: Attributes are selected based on information-theoretic
   measures. The most informative attributes are tested first in the tree.

4. TREE SIZE: Larger training sets generally produce larger trees, but not
   always. The tree size depends on the complexity of the decision boundary.

5. GREEDY ALGORITHM: ID3 makes locally optimal choices at each node. It
   doesn't backtrack, so may miss globally optimal trees (Section 4).

Extensions discussed in paper (Sections 5-7):
- Chi-square test to handle noisy data
- Probabilistic methods for unknown attribute values
- Gain ratio criterion to reduce bias toward multi-valued attributes
    """)

    print("="*70)


if __name__ == "__main__":
    main()
