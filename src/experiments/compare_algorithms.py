"""
Compare ID3 and C4.5 algorithms on UCI ML datasets.

This script runs both algorithms on the same test data and provides
detailed comparisons of their performance, tree complexity, and behavior.
"""

import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
import time
from typing import List, Dict, Any, Tuple
from collections import Counter
from src.algorithms import ID3
from src.algorithms import C45
from src.datasets import (download_mushroom_dataset, download_tic_tac_toe_dataset,
                          download_voting_dataset, train_test_split, evaluate_model)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def count_nodes(node) -> int:
    """Count total nodes in a tree."""
    if node is None:
        return 0
    if node.is_leaf():
        return 1
    return 1 + sum(count_nodes(child) for child in node.children.values())


def count_leaves(node) -> int:
    """Count leaf nodes in a tree."""
    if node is None:
        return 0
    if node.is_leaf():
        return 1
    return sum(count_leaves(child) for child in node.children.values())


def tree_depth(node) -> int:
    """Calculate maximum depth of tree."""
    if node is None or node.is_leaf():
        return 0
    if not node.children:
        return 0
    return 1 + max(tree_depth(child) for child in node.children.values())


def compare_on_dataset(name: str, examples: List[Dict[str, Any]],
                       class_attr: str = 'class',
                       continuous_attrs: List[str] = None) -> Dict[str, Any]:
    """
    Compare ID3 and C4.5 on a single dataset.

    Args:
        name: Dataset name
        examples: All examples
        class_attr: Name of class attribute
        continuous_attrs: List of continuous attribute names

    Returns:
        Dictionary with comparison results
    """
    if not examples:
        return None

    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")

    # Split data (70/30 split)
    random.seed(42)
    train_set, test_set = train_test_split(examples, test_ratio=0.3)

    print(f"Dataset size: {len(examples)} examples")
    print(f"Training set: {len(train_set)} examples")
    print(f"Test set:     {len(test_set)} examples")

    # Count classes
    class_counts = Counter(ex[class_attr] for ex in examples)
    print(f"Classes: {class_counts}")

    results = {
        'name': name,
        'total_examples': len(examples),
        'train_size': len(train_set),
        'test_size': len(test_set),
        'n_classes': len(class_counts),
        'class_distribution': dict(class_counts)
    }

    # Train and evaluate ID3
    print(f"\n{'ID3 (1986)':^35}{'C4.5 (1993)':^35}")
    print("-" * 70)

    try:
        # ID3
        start_time = time.time()
        id3_model = ID3()
        id3_model.fit(train_set, class_attr=class_attr)
        id3_train_time = time.time() - start_time

        id3_train_acc = evaluate_model(id3_model, train_set, class_attr)
        id3_test_acc = evaluate_model(id3_model, test_set, class_attr)
        id3_nodes = count_nodes(id3_model.root)
        id3_leaves = count_leaves(id3_model.root)
        id3_depth = tree_depth(id3_model.root)

        results['id3'] = {
            'train_accuracy': id3_train_acc,
            'test_accuracy': id3_test_acc,
            'nodes': id3_nodes,
            'leaves': id3_leaves,
            'depth': id3_depth,
            'train_time': id3_train_time,
            'overfitting': id3_train_acc - id3_test_acc
        }

    except Exception as e:
        print(f"ID3 error: {e}")
        results['id3'] = None

    try:
        # C4.5 with pruning
        start_time = time.time()
        c45_model = C45(pruning=True)
        c45_model.fit(train_set, class_attr=class_attr, continuous_attrs=continuous_attrs)
        c45_train_time = time.time() - start_time

        c45_train_acc = evaluate_model(c45_model, train_set, class_attr)
        c45_test_acc = evaluate_model(c45_model, test_set, class_attr)
        c45_nodes = count_nodes(c45_model.root)
        c45_leaves = count_leaves(c45_model.root)
        c45_depth = tree_depth(c45_model.root)

        results['c45'] = {
            'train_accuracy': c45_train_acc,
            'test_accuracy': c45_test_acc,
            'nodes': c45_nodes,
            'leaves': c45_leaves,
            'depth': c45_depth,
            'train_time': c45_train_time,
            'overfitting': c45_train_acc - c45_test_acc
        }

    except Exception as e:
        print(f"C4.5 error: {e}")
        results['c45'] = None

    # Print comparison table
    if results['id3'] and results['c45']:
        metrics = [
            ('Training Accuracy', f"{results['id3']['train_accuracy']*100:.2f}%",
             f"{results['c45']['train_accuracy']*100:.2f}%"),
            ('Test Accuracy', f"{results['id3']['test_accuracy']*100:.2f}%",
             f"{results['c45']['test_accuracy']*100:.2f}%"),
            ('Overfitting Gap', f"{results['id3']['overfitting']*100:.2f}%",
             f"{results['c45']['overfitting']*100:.2f}%"),
            ('Tree Nodes', f"{results['id3']['nodes']}",
             f"{results['c45']['nodes']}"),
            ('Leaf Nodes', f"{results['id3']['leaves']}",
             f"{results['c45']['leaves']}"),
            ('Tree Depth', f"{results['id3']['depth']}",
             f"{results['c45']['depth']}"),
            ('Training Time', f"{results['id3']['train_time']:.3f}s",
             f"{results['c45']['train_time']:.3f}s"),
        ]

        for metric, id3_val, c45_val in metrics:
            print(f"{metric:<20} {id3_val:>15} {c45_val:>15}")

        # Determine winner
        print(f"\n{'Analysis':-^70}")

        # Test accuracy comparison
        if results['c45']['test_accuracy'] > results['id3']['test_accuracy']:
            improvement = (results['c45']['test_accuracy'] - results['id3']['test_accuracy']) * 100
            print(f"✓ C4.5 has {improvement:.2f}% better test accuracy")
        elif results['id3']['test_accuracy'] > results['c45']['test_accuracy']:
            diff = (results['id3']['test_accuracy'] - results['c45']['test_accuracy']) * 100
            print(f"✓ ID3 has {diff:.2f}% better test accuracy")
        else:
            print(f"= Both achieve same test accuracy")

        # Tree complexity comparison
        size_reduction = (1 - results['c45']['nodes'] / results['id3']['nodes']) * 100
        if results['c45']['nodes'] < results['id3']['nodes']:
            print(f"✓ C4.5 tree is {size_reduction:.1f}% smaller ({results['c45']['nodes']} vs {results['id3']['nodes']} nodes)")
        elif results['c45']['nodes'] > results['id3']['nodes']:
            print(f"✗ C4.5 tree is larger ({results['c45']['nodes']} vs {results['id3']['nodes']} nodes)")
        else:
            print(f"= Both have same tree size")

        # Overfitting comparison
        if results['c45']['overfitting'] < results['id3']['overfitting']:
            print(f"✓ C4.5 has less overfitting ({results['c45']['overfitting']*100:.2f}% vs {results['id3']['overfitting']*100:.2f}%)")
        elif results['c45']['overfitting'] > results['id3']['overfitting']:
            print(f"✗ C4.5 has more overfitting ({results['c45']['overfitting']*100:.2f}% vs {results['id3']['overfitting']*100:.2f}%)")
        else:
            print(f"= Both have same overfitting level")

    return results


def plot_comparison(all_results: List[Dict[str, Any]], output_file: str = 'comparison_plot.png'):
    """Create comparison visualization."""
    if not HAS_MATPLOTLIB:
        print("\n⚠️  Matplotlib not available for plotting.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    dataset_names = [r['name'] for r in all_results if r and r['id3'] and r['c45']]

    # Plot 1: Test Accuracy
    id3_accs = [r['id3']['test_accuracy'] * 100 for r in all_results if r and r['id3'] and r['c45']]
    c45_accs = [r['c45']['test_accuracy'] * 100 for r in all_results if r and r['id3'] and r['c45']]

    x = range(len(dataset_names))
    width = 0.35

    axes[0, 0].bar([i - width/2 for i in x], id3_accs, width, label='ID3', alpha=0.8, color='steelblue')
    axes[0, 0].bar([i + width/2 for i in x], c45_accs, width, label='C4.5', alpha=0.8, color='darkorange')
    axes[0, 0].set_ylabel('Test Accuracy (%)', fontsize=11)
    axes[0, 0].set_title('Test Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(dataset_names, rotation=15, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([min(min(id3_accs), min(c45_accs)) - 5, 105])

    # Plot 2: Tree Size (Nodes)
    id3_nodes = [r['id3']['nodes'] for r in all_results if r and r['id3'] and r['c45']]
    c45_nodes = [r['c45']['nodes'] for r in all_results if r and r['id3'] and r['c45']]

    axes[0, 1].bar([i - width/2 for i in x], id3_nodes, width, label='ID3', alpha=0.8, color='steelblue')
    axes[0, 1].bar([i + width/2 for i in x], c45_nodes, width, label='C4.5', alpha=0.8, color='darkorange')
    axes[0, 1].set_ylabel('Number of Nodes', fontsize=11)
    axes[0, 1].set_title('Tree Complexity (Size)', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(dataset_names, rotation=15, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Overfitting (Train - Test accuracy)
    id3_overfit = [r['id3']['overfitting'] * 100 for r in all_results if r and r['id3'] and r['c45']]
    c45_overfit = [r['c45']['overfitting'] * 100 for r in all_results if r and r['id3'] and r['c45']]

    axes[1, 0].bar([i - width/2 for i in x], id3_overfit, width, label='ID3', alpha=0.8, color='steelblue')
    axes[1, 0].bar([i + width/2 for i in x], c45_overfit, width, label='C4.5', alpha=0.8, color='darkorange')
    axes[1, 0].set_ylabel('Overfitting Gap (%)', fontsize=11)
    axes[1, 0].set_title('Overfitting: Train Acc - Test Acc', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(dataset_names, rotation=15, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Plot 4: Tree Depth
    id3_depth = [r['id3']['depth'] for r in all_results if r and r['id3'] and r['c45']]
    c45_depth = [r['c45']['depth'] for r in all_results if r and r['id3'] and r['c45']]

    axes[1, 1].bar([i - width/2 for i in x], id3_depth, width, label='ID3', alpha=0.8, color='steelblue')
    axes[1, 1].bar([i + width/2 for i in x], c45_depth, width, label='C4.5', alpha=0.8, color='darkorange')
    axes[1, 1].set_ylabel('Tree Depth', fontsize=11)
    axes[1, 1].set_title('Tree Complexity (Depth)', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(dataset_names, rotation=15, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: {output_file}")


def print_summary(all_results: List[Dict[str, Any]]):
    """Print overall summary of comparisons."""
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}\n")

    valid_results = [r for r in all_results if r and r['id3'] and r['c45']]

    if not valid_results:
        print("No valid results to summarize.")
        return

    # Count wins
    c45_accuracy_wins = 0
    c45_size_wins = 0
    c45_overfit_wins = 0
    ties_accuracy = 0
    ties_size = 0

    for r in valid_results:
        # Accuracy wins
        if r['c45']['test_accuracy'] > r['id3']['test_accuracy']:
            c45_accuracy_wins += 1
        elif r['c45']['test_accuracy'] == r['id3']['test_accuracy']:
            ties_accuracy += 1

        # Size wins (smaller is better)
        if r['c45']['nodes'] < r['id3']['nodes']:
            c45_size_wins += 1
        elif r['c45']['nodes'] == r['id3']['nodes']:
            ties_size += 1

        # Overfitting wins (less is better)
        if r['c45']['overfitting'] < r['id3']['overfitting']:
            c45_overfit_wins += 1

    total = len(valid_results)

    print(f"Datasets tested: {total}")
    print(f"\n{'Metric':<30} {'C4.5 Wins':<15} {'Ties':<15} {'ID3 Wins':<15}")
    print("-" * 70)
    print(f"{'Test Accuracy':<30} {c45_accuracy_wins:<15} {ties_accuracy:<15} {total - c45_accuracy_wins - ties_accuracy:<15}")
    print(f"{'Tree Size (smaller better)':<30} {c45_size_wins:<15} {ties_size:<15} {total - c45_size_wins - ties_size:<15}")
    print(f"{'Less Overfitting':<30} {c45_overfit_wins:<15} {0:<15} {total - c45_overfit_wins:<15}")

    # Calculate average improvements
    avg_acc_improvement = sum((r['c45']['test_accuracy'] - r['id3']['test_accuracy']) * 100
                              for r in valid_results) / total
    avg_size_reduction = sum((1 - r['c45']['nodes'] / r['id3']['nodes']) * 100
                            for r in valid_results if r['id3']['nodes'] > 0) / total
    avg_overfit_reduction = sum((r['id3']['overfitting'] - r['c45']['overfitting']) * 100
                               for r in valid_results) / total

    print(f"\n{'Average Changes (C4.5 vs ID3)':^70}")
    print("-" * 70)
    print(f"Test accuracy change:  {avg_acc_improvement:+.2f}%")
    print(f"Tree size change:      {-avg_size_reduction:+.2f}%")
    print(f"Overfitting reduction: {avg_overfit_reduction:+.2f}%")

    print(f"\n{'Key Insights':^70}")
    print("-" * 70)
    print("""
C4.5 Improvements over ID3:

1. GAIN RATIO CRITERION: Reduces bias toward multi-valued attributes,
   leading to better attribute selection and often simpler trees.

2. PRUNING: Post-pruning reduces overfitting by removing branches that
   don't improve generalization, resulting in smaller trees.

3. CONTINUOUS ATTRIBUTES: Can handle numeric attributes directly without
   manual discretization, finding optimal split points automatically.

4. MISSING VALUES: Better handling of missing attribute values using
   probabilistic approaches.

5. COMPUTATIONAL COST: Slightly higher training time due to pruning and
   more sophisticated attribute selection, but often worth the trade-off.
    """)


def main():
    """Run comprehensive comparison."""
    print("="*70)
    print("ID3 (1986) vs C4.5 (1993) - Comprehensive Comparison")
    print("="*70)
    print("\nComparing Quinlan's algorithms on UCI ML Repository datasets...")

    all_results = []

    # Test Mushroom dataset
    mushroom_data = download_mushroom_dataset()
    if mushroom_data:
        result = compare_on_dataset("Mushroom Classification", mushroom_data,
                                   class_attr='class')
        all_results.append(result)

    # Test Tic-Tac-Toe dataset
    tictactoe_data = download_tic_tac_toe_dataset()
    if tictactoe_data:
        result = compare_on_dataset("Tic-Tac-Toe Endgame", tictactoe_data,
                                   class_attr='class')
        all_results.append(result)

    # Test Voting dataset
    voting_data = download_voting_dataset()
    if voting_data:
        result = compare_on_dataset("Congressional Voting", voting_data,
                                   class_attr='class')
        all_results.append(result)

    # Generate visualizations
    plot_comparison(all_results)

    # Print summary
    print_summary(all_results)

    print("\n" + "="*70)
    print("Comparison complete!")
    print("="*70)


if __name__ == "__main__":
    main()
