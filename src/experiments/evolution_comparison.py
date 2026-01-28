"""
Evolution of Decision Tree Algorithms: ID3 → C4.5 → XGBoost

Comprehensive comparison showing 40 years of algorithm evolution:
- ID3 (1986): Quinlan's foundational information-theoretic approach
- C4.5 (1993): Enhanced with pruning and gain ratio
- XGBoost (2014): Modern gradient boosting champion

This script demonstrates the massive improvements in modern ML.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
import time
import warnings
from typing import List, Dict, Any, Tuple
from collections import Counter

import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np

from src.algorithms import ID3
from src.algorithms import C45
from src.datasets import (download_mushroom_dataset, download_tic_tac_toe_dataset,
                          download_voting_dataset, train_test_split, evaluate_model)

warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def prepare_data_for_xgboost(examples: List[Dict[str, Any]],
                             class_attr: str = 'class') -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert dictionary-based examples to numpy arrays for XGBoost.

    XGBoost requires numeric features, so we encode categorical variables.
    """
    if not examples:
        return np.array([]), np.array([])

    # Get all attributes except class
    attributes = [attr for attr in examples[0].keys() if attr != class_attr]

    # Build encoding dictionaries for categorical features
    encodings = {}
    for attr in attributes:
        unique_vals = list(set(ex.get(attr, '?') for ex in examples))
        encodings[attr] = {val: i for i, val in enumerate(unique_vals)}

    # Encode class labels
    unique_classes = list(set(ex[class_attr] for ex in examples))
    class_encoding = {cls: i for i, cls in enumerate(unique_classes)}

    # Convert to numeric arrays
    X = []
    y = []

    for ex in examples:
        features = []
        for attr in attributes:
            val = ex.get(attr, '?')
            features.append(encodings[attr].get(val, -1))
        X.append(features)
        y.append(class_encoding[ex[class_attr]])

    return np.array(X), np.array(y), class_encoding


def compare_three_algorithms(name: str, examples: List[Dict[str, Any]],
                            class_attr: str = 'class') -> Dict[str, Any]:
    """
    Compare ID3, C4.5, and XGBoost on a single dataset.

    Returns comprehensive metrics for all three algorithms.
    """
    if not examples:
        return None

    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")

    # Split data
    random.seed(42)
    train_set, test_set = train_test_split(examples, test_ratio=0.3)

    print(f"Dataset: {len(examples)} examples")
    print(f"Training: {len(train_set)} | Test: {len(test_set)}")

    # Count classes
    class_counts = Counter(ex[class_attr] for ex in examples)
    print(f"Classes: {dict(class_counts)}")

    results = {
        'name': name,
        'total_examples': len(examples),
        'train_size': len(train_set),
        'test_size': len(test_set),
    }

    # Headers
    print(f"\n{'Algorithm':<20} {'Test Acc':<12} {'Train Time':<12} {'Complexity':<15}")
    print("-" * 80)

    # ========== ID3 (1986) ==========
    try:
        start = time.time()
        id3_model = ID3()
        id3_model.fit(train_set, class_attr=class_attr)
        id3_time = time.time() - start

        id3_train_acc = evaluate_model(id3_model, train_set, class_attr)
        id3_test_acc = evaluate_model(id3_model, test_set, class_attr)

        def count_nodes(node):
            if node is None or node.is_leaf():
                return 1
            return 1 + sum(count_nodes(child) for child in node.children.values())

        id3_nodes = count_nodes(id3_model.root)

        results['id3'] = {
            'train_accuracy': id3_train_acc,
            'test_accuracy': id3_test_acc,
            'train_time': id3_time,
            'nodes': id3_nodes,
            'overfitting': id3_train_acc - id3_test_acc
        }

        print(f"{'ID3 (1986)':<20} {id3_test_acc*100:>10.2f}% {id3_time:>10.3f}s {id3_nodes:>10} nodes")

    except Exception as e:
        print(f"{'ID3 (1986)':<20} {'ERROR':<12}")
        results['id3'] = None

    # ========== C4.5 (1993) ==========
    try:
        start = time.time()
        c45_model = C45(pruning=True)
        c45_model.fit(train_set, class_attr=class_attr)
        c45_time = time.time() - start

        c45_train_acc = evaluate_model(c45_model, train_set, class_attr)
        c45_test_acc = evaluate_model(c45_model, test_set, class_attr)

        def count_nodes_c45(node):
            if node is None or node.is_leaf():
                return 1
            return 1 + sum(count_nodes_c45(child) for child in node.children.values())

        c45_nodes = count_nodes_c45(c45_model.root)

        results['c45'] = {
            'train_accuracy': c45_train_acc,
            'test_accuracy': c45_test_acc,
            'train_time': c45_time,
            'nodes': c45_nodes,
            'overfitting': c45_train_acc - c45_test_acc
        }

        print(f"{'C4.5 (1993)':<20} {c45_test_acc*100:>10.2f}% {c45_time:>10.3f}s {c45_nodes:>10} nodes")

    except Exception as e:
        print(f"{'C4.5 (1993)':<20} {'ERROR':<12}")
        results['c45'] = None

    # ========== XGBoost (2014) ==========
    try:
        # Prepare data for XGBoost
        X_train, y_train, class_encoding = prepare_data_for_xgboost(train_set, class_attr)
        X_test, y_test, _ = prepare_data_for_xgboost(test_set, class_attr)

        # Train XGBoost
        start = time.time()

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Parameters optimized for comparison
        params = {
            'max_depth': 6,
            'eta': 0.3,
            'objective': 'binary:logistic' if len(class_encoding) == 2 else 'multi:softmax',
            'num_class': len(class_encoding) if len(class_encoding) > 2 else None,
            'eval_metric': 'error',
            'seed': 42
        }

        # Remove num_class if binary
        if len(class_encoding) == 2:
            del params['num_class']

        # Train
        num_rounds = 100
        xgb_model = xgb.train(
            params,
            dtrain,
            num_rounds,
            evals=[(dtest, 'test')],
            verbose_eval=False
        )

        xgb_time = time.time() - start

        # Predict
        train_preds = xgb_model.predict(dtrain)
        test_preds = xgb_model.predict(dtest)

        # For binary classification, convert probabilities to classes
        if len(class_encoding) == 2:
            train_preds = (train_preds > 0.5).astype(int)
            test_preds = (test_preds > 0.5).astype(int)
        else:
            train_preds = train_preds.astype(int)
            test_preds = test_preds.astype(int)

        xgb_train_acc = accuracy_score(y_train, train_preds)
        xgb_test_acc = accuracy_score(y_test, test_preds)

        # Get number of trees
        num_trees = len(xgb_model.get_dump())

        results['xgboost'] = {
            'train_accuracy': xgb_train_acc,
            'test_accuracy': xgb_test_acc,
            'train_time': xgb_time,
            'num_trees': num_trees,
            'num_rounds': num_rounds,
            'overfitting': xgb_train_acc - xgb_test_acc
        }

        print(f"{'XGBoost (2014)':<20} {xgb_test_acc*100:>10.2f}% {xgb_time:>10.3f}s {num_trees:>7} trees")

    except Exception as e:
        print(f"{'XGBoost (2014)':<20} {'ERROR: ' + str(e):<12}")
        results['xgboost'] = None

    # Analysis
    print(f"\n{'Analysis':-^80}")

    if results['id3'] and results['c45'] and results['xgboost']:
        # Best test accuracy
        accs = {
            'ID3': results['id3']['test_accuracy'],
            'C4.5': results['c45']['test_accuracy'],
            'XGBoost': results['xgboost']['test_accuracy']
        }
        best_algo = max(accs, key=accs.get)
        best_acc = accs[best_algo]

        print(f"\n🏆 Best Test Accuracy: {best_algo} ({best_acc*100:.2f}%)")

        # Improvements from ID3
        c45_improvement = (results['c45']['test_accuracy'] - results['id3']['test_accuracy']) * 100
        xgb_improvement = (results['xgboost']['test_accuracy'] - results['id3']['test_accuracy']) * 100

        print(f"\nImprovements over ID3:")
        print(f"  C4.5:    {c45_improvement:+.2f}%")
        print(f"  XGBoost: {xgb_improvement:+.2f}%")

        # Overfitting comparison
        print(f"\nOverfitting (Train - Test):")
        print(f"  ID3:     {results['id3']['overfitting']*100:.2f}%")
        print(f"  C4.5:    {results['c45']['overfitting']*100:.2f}%")
        print(f"  XGBoost: {results['xgboost']['overfitting']*100:.2f}%")

        # Speed comparison
        print(f"\nTraining Time:")
        print(f"  ID3:     {results['id3']['train_time']:.3f}s (baseline)")
        print(f"  C4.5:    {results['c45']['train_time']:.3f}s ({results['c45']['train_time']/results['id3']['train_time']:.1f}x)")
        print(f"  XGBoost: {results['xgboost']['train_time']:.3f}s ({results['xgboost']['train_time']/results['id3']['train_time']:.1f}x)")

    return results


def plot_evolution(all_results: List[Dict[str, Any]], output_file: str = 'evolution_plot.png'):
    """Create comprehensive evolution visualization."""
    if not HAS_MATPLOTLIB:
        print("\n⚠️  Matplotlib not available for plotting.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    dataset_names = [r['name'] for r in all_results if r and r.get('id3') and r.get('c45') and r.get('xgboost')]

    # Plot 1: Test Accuracy Evolution
    id3_accs = [r['id3']['test_accuracy'] * 100 for r in all_results if r and r.get('id3') and r.get('c45') and r.get('xgboost')]
    c45_accs = [r['c45']['test_accuracy'] * 100 for r in all_results if r and r.get('id3') and r.get('c45') and r.get('xgboost')]
    xgb_accs = [r['xgboost']['test_accuracy'] * 100 for r in all_results if r and r.get('id3') and r.get('c45') and r.get('xgboost')]

    x = range(len(dataset_names))
    width = 0.25

    axes[0, 0].bar([i - width for i in x], id3_accs, width, label='ID3 (1986)', alpha=0.8, color='#5DA5DA')
    axes[0, 0].bar(x, c45_accs, width, label='C4.5 (1993)', alpha=0.8, color='#FAA43A')
    axes[0, 0].bar([i + width for i in x], xgb_accs, width, label='XGBoost (2014)', alpha=0.8, color='#60BD68')
    axes[0, 0].set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Evolution of Test Accuracy (1986 → 2014)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(dataset_names, rotation=15, ha='right')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([min(min(id3_accs), min(c45_accs), min(xgb_accs)) - 5, 105])

    # Plot 2: Overfitting Reduction
    id3_overfit = [r['id3']['overfitting'] * 100 for r in all_results if r and r.get('id3') and r.get('c45') and r.get('xgboost')]
    c45_overfit = [r['c45']['overfitting'] * 100 for r in all_results if r and r.get('id3') and r.get('c45') and r.get('xgboost')]
    xgb_overfit = [r['xgboost']['overfitting'] * 100 for r in all_results if r and r.get('id3') and r.get('c45') and r.get('xgboost')]

    axes[0, 1].bar([i - width for i in x], id3_overfit, width, label='ID3 (1986)', alpha=0.8, color='#5DA5DA')
    axes[0, 1].bar(x, c45_overfit, width, label='C4.5 (1993)', alpha=0.8, color='#FAA43A')
    axes[0, 1].bar([i + width for i in x], xgb_overfit, width, label='XGBoost (2014)', alpha=0.8, color='#60BD68')
    axes[0, 1].set_ylabel('Overfitting Gap (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Overfitting Reduction Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(dataset_names, rotation=15, ha='right')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Plot 3: Training Time Comparison
    id3_time = [r['id3']['train_time'] for r in all_results if r and r.get('id3') and r.get('c45') and r.get('xgboost')]
    c45_time = [r['c45']['train_time'] for r in all_results if r and r.get('id3') and r.get('c45') and r.get('xgboost')]
    xgb_time = [r['xgboost']['train_time'] for r in all_results if r and r.get('id3') and r.get('c45') and r.get('xgboost')]

    axes[1, 0].bar([i - width for i in x], id3_time, width, label='ID3 (1986)', alpha=0.8, color='#5DA5DA')
    axes[1, 0].bar(x, c45_time, width, label='C4.5 (1993)', alpha=0.8, color='#FAA43A')
    axes[1, 0].bar([i + width for i in x], xgb_time, width, label='XGBoost (2014)', alpha=0.8, color='#60BD68')
    axes[1, 0].set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Computational Cost', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(dataset_names, rotation=15, ha='right')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_yscale('log')

    # Plot 4: Relative Improvements
    c45_improvements = [(r['c45']['test_accuracy'] - r['id3']['test_accuracy']) * 100
                       for r in all_results if r and r.get('id3') and r.get('c45') and r.get('xgboost')]
    xgb_improvements = [(r['xgboost']['test_accuracy'] - r['id3']['test_accuracy']) * 100
                       for r in all_results if r and r.get('id3') and r.get('c45') and r.get('xgboost')]

    axes[1, 1].bar([i - width/2 for i in x], c45_improvements, width,
                   label='C4.5 vs ID3', alpha=0.8, color='#FAA43A')
    axes[1, 1].bar([i + width/2 for i in x], xgb_improvements, width,
                   label='XGBoost vs ID3', alpha=0.8, color='#60BD68')
    axes[1, 1].set_ylabel('Accuracy Improvement (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Improvements Over ID3 Baseline', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(dataset_names, rotation=15, ha='right')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    plt.suptitle('40 Years of Decision Tree Evolution: ID3 → C4.5 → XGBoost',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Evolution plot saved to: {output_file}")


def print_summary(all_results: List[Dict[str, Any]]):
    """Print comprehensive summary of the evolution."""
    print(f"\n{'='*80}")
    print("40 YEARS OF ALGORITHM EVOLUTION: SUMMARY")
    print(f"{'='*80}\n")

    valid_results = [r for r in all_results if r and r.get('id3') and r.get('c45') and r.get('xgboost')]

    if not valid_results:
        print("No valid results to summarize.")
        return

    # Calculate averages
    avg_id3_acc = np.mean([r['id3']['test_accuracy'] for r in valid_results])
    avg_c45_acc = np.mean([r['c45']['test_accuracy'] for r in valid_results])
    avg_xgb_acc = np.mean([r['xgboost']['test_accuracy'] for r in valid_results])

    avg_id3_time = np.mean([r['id3']['train_time'] for r in valid_results])
    avg_c45_time = np.mean([r['c45']['train_time'] for r in valid_results])
    avg_xgb_time = np.mean([r['xgboost']['train_time'] for r in valid_results])

    avg_id3_overfit = np.mean([r['id3']['overfitting'] for r in valid_results])
    avg_c45_overfit = np.mean([r['c45']['overfitting'] for r in valid_results])
    avg_xgb_overfit = np.mean([r['xgboost']['overfitting'] for r in valid_results])

    print(f"{'Algorithm':<20} {'Avg Test Acc':<15} {'Avg Train Time':<15} {'Avg Overfitting':<15}")
    print("-" * 80)
    print(f"{'ID3 (1986)':<20} {avg_id3_acc*100:>13.2f}% {avg_id3_time:>13.3f}s {avg_id3_overfit*100:>13.2f}%")
    print(f"{'C4.5 (1993)':<20} {avg_c45_acc*100:>13.2f}% {avg_c45_time:>13.3f}s {avg_c45_overfit*100:>13.2f}%")
    print(f"{'XGBoost (2014)':<20} {avg_xgb_acc*100:>13.2f}% {avg_xgb_time:>13.3f}s {avg_xgb_overfit*100:>13.2f}%")

    print(f"\n{'Improvements Over ID3':-^80}")
    c45_acc_gain = (avg_c45_acc - avg_id3_acc) * 100
    xgb_acc_gain = (avg_xgb_acc - avg_id3_acc) * 100

    print(f"\nTest Accuracy:")
    print(f"  C4.5:    {c45_acc_gain:+.2f}% improvement")
    print(f"  XGBoost: {xgb_acc_gain:+.2f}% improvement")

    print(f"\nOverfitting Reduction:")
    c45_overfit_reduction = (avg_id3_overfit - avg_c45_overfit) * 100
    xgb_overfit_reduction = (avg_id3_overfit - avg_xgb_overfit) * 100
    print(f"  C4.5:    {c45_overfit_reduction:+.2f}% less overfitting")
    print(f"  XGBoost: {xgb_overfit_reduction:+.2f}% less overfitting")

    print(f"\nComputational Cost:")
    print(f"  C4.5:    {avg_c45_time/avg_id3_time:.1f}x slower than ID3")
    print(f"  XGBoost: {avg_xgb_time/avg_id3_time:.1f}x slower than ID3")

    print(f"\n{'Key Insights':-^80}")
    print("""
1. ACCURACY EVOLUTION: Each generation brings measurable improvements
   - C4.5 (1993): Modest gains through pruning and gain ratio
   - XGBoost (2014): Significant leap through ensemble learning

2. OVERFITTING CONTROL: Modern algorithms handle overfitting much better
   - ID3: No mechanism, relies on perfect training accuracy
   - C4.5: Pruning reduces overfitting
   - XGBoost: Regularization + boosting provides best generalization

3. COMPUTATIONAL TRADE-OFF: More sophisticated = more computation
   - XGBoost is slowest but most accurate
   - Worth the cost for production systems

4. WHEN TO USE WHAT:
   - ID3: Educational purposes, understanding fundamentals
   - C4.5: When interpretability is crucial, simple problems
   - XGBoost: When accuracy matters most, production systems

5. THE GRADIENT BOOSTING REVOLUTION (2014-present):
   - XGBoost dominates Kaggle competitions
   - Ensemble of weak learners beats single strong learner
   - Regularization prevents overfitting
   - Feature importance for interpretability
    """)


def main():
    """Run comprehensive three-way comparison."""
    print("="*80)
    print("EVOLUTION OF DECISION TREES: ID3 (1986) → C4.5 (1993) → XGBoost (2014)")
    print("="*80)
    print("\nShowing 40 years of machine learning progress on the same datasets...")

    all_results = []

    # Test Mushroom dataset
    print("\n[1/3] Loading datasets...")
    mushroom_data = download_mushroom_dataset()
    if mushroom_data:
        result = compare_three_algorithms("Mushroom Classification", mushroom_data)
        all_results.append(result)

    # Test Tic-Tac-Toe dataset
    tictactoe_data = download_tic_tac_toe_dataset()
    if tictactoe_data:
        result = compare_three_algorithms("Tic-Tac-Toe Endgame", tictactoe_data)
        all_results.append(result)

    # Test Voting dataset
    voting_data = download_voting_dataset()
    if voting_data:
        result = compare_three_algorithms("Congressional Voting", voting_data)
        all_results.append(result)

    # Generate visualization
    print("\n[2/3] Generating evolution visualization...")
    plot_evolution(all_results)

    # Print summary
    print("\n[3/3] Analyzing results...")
    print_summary(all_results)

    print("\n" + "="*80)
    print("Evolution comparison complete!")
    print("="*80)
    print("\nFiles generated:")
    print("  - evolution_plot.png: 4-panel visualization of algorithm evolution")
    print("\nNext steps:")
    print("  - View the plot to see the dramatic improvements")
    print("  - Read EVOLUTION.md for detailed analysis (to be created)")


if __name__ == "__main__":
    main()
