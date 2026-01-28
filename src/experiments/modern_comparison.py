"""
Compare all algorithms including modern state-of-the-art:
ID3 (1986) → C4.5 (1993) → XGBoost (2014) → LightGBM (2017)

Shows that gradient boosting evolution continues beyond XGBoost.
"""

import sys
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.algorithms import ID3
from src.algorithms import C45
import xgboost as xgb
import lightgbm as lgb
from src.datasets import download_mushroom_dataset, download_tic_tac_toe_dataset, download_voting_dataset
import numpy as np
import time
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def prepare_data_for_boosting(examples, class_attr='class'):
    """Convert dictionary examples to arrays for gradient boosting."""
    if not examples:
        return None, None, None, None

    # Get attributes
    attributes = [k for k in examples[0].keys() if k != class_attr]

    # Create encodings
    encodings = {}
    for attr in attributes:
        unique_vals = list(set(ex.get(attr, '?') for ex in examples))
        encodings[attr] = {val: i for i, val in enumerate(unique_vals)}

    # Encode data
    X = []
    y = []
    class_values = list(set(ex[class_attr] for ex in examples))
    class_encoding = {val: i for i, val in enumerate(class_values)}

    for ex in examples:
        features = [encodings[attr].get(ex.get(attr, '?'), 0) for attr in attributes]
        X.append(features)
        y.append(class_encoding[ex[class_attr]])

    return np.array(X), np.array(y), class_encoding, attributes


def evaluate_model(model, examples, class_attr='class'):
    """Evaluate accuracy."""
    if not examples:
        return 0.0
    correct = sum(1 for ex in examples if model.predict(ex) == ex[class_attr])
    return 100.0 * correct / len(examples)


def compare_all_algorithms(name, examples, class_attr='class'):
    """Compare ID3, C4.5, XGBoost, and LightGBM."""

    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")

    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(examples))
    train_size = int(0.7 * len(examples))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_set = [examples[i] for i in train_indices]
    test_set = [examples[i] for i in test_indices]

    print(f"Dataset: {len(examples)} examples")
    print(f"Training: {len(train_set)} | Test: {len(test_set)}")

    class_counts = Counter(ex[class_attr] for ex in examples)
    print(f"Classes: {dict(class_counts)}\n")

    results = {}

    # === ID3 (1986) ===
    print(f"[1/4] Training ID3 (1986)...")
    try:
        start_time = time.time()
        id3_model = ID3()
        id3_model.fit(train_set, class_attr=class_attr)
        id3_train_time = time.time() - start_time

        id3_train_acc = evaluate_model(id3_model, train_set, class_attr)
        id3_test_acc = evaluate_model(id3_model, test_set, class_attr)
        id3_overfitting = id3_train_acc - id3_test_acc

        def count_nodes(node):
            if node is None or node.is_leaf():
                return 1 if node else 0
            return 1 + sum(count_nodes(c) for c in node.children.values())

        id3_nodes = count_nodes(id3_model.root)

        results['id3'] = {
            'train_acc': id3_train_acc,
            'test_acc': id3_test_acc,
            'overfitting': id3_overfitting,
            'train_time': id3_train_time,
            'complexity': f"{id3_nodes} nodes"
        }
        print(f"  ✓ Test Accuracy: {id3_test_acc:.2f}% | Time: {id3_train_time:.3f}s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['id3'] = None

    # === C4.5 (1993) ===
    print(f"[2/4] Training C4.5 (1993)...")
    try:
        start_time = time.time()
        c45_model = C45(pruning=True, confidence_level=0.25)
        c45_model.fit(train_set, class_attr=class_attr)
        c45_train_time = time.time() - start_time

        c45_train_acc = evaluate_model(c45_model, train_set, class_attr)
        c45_test_acc = evaluate_model(c45_model, test_set, class_attr)
        c45_overfitting = c45_train_acc - c45_test_acc

        c45_nodes = count_nodes(c45_model.root)

        results['c45'] = {
            'train_acc': c45_train_acc,
            'test_acc': c45_test_acc,
            'overfitting': c45_overfitting,
            'train_time': c45_train_time,
            'complexity': f"{c45_nodes} nodes"
        }
        print(f"  ✓ Test Accuracy: {c45_test_acc:.2f}% | Time: {c45_train_time:.3f}s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['c45'] = None

    # === XGBoost (2014) ===
    print(f"[3/4] Training XGBoost (2014)...")
    try:
        X_train, y_train, class_encoding, attributes = prepare_data_for_boosting(train_set, class_attr)
        X_test, y_test, _, _ = prepare_data_for_boosting(test_set, class_attr)

        start_time = time.time()
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=attributes)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=attributes)

        params = {
            'max_depth': 6,
            'eta': 0.3,
            'objective': 'binary:logistic',
            'eval_metric': 'error',
            'seed': 42
        }
        xgb_model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
        xgb_train_time = time.time() - start_time

        # Evaluate
        train_preds = (xgb_model.predict(dtrain) > 0.5).astype(int)
        test_preds = (xgb_model.predict(dtest) > 0.5).astype(int)

        xgb_train_acc = 100.0 * np.mean(train_preds == y_train)
        xgb_test_acc = 100.0 * np.mean(test_preds == y_test)
        xgb_overfitting = xgb_train_acc - xgb_test_acc

        results['xgboost'] = {
            'train_acc': xgb_train_acc,
            'test_acc': xgb_test_acc,
            'overfitting': xgb_overfitting,
            'train_time': xgb_train_time,
            'complexity': '100 trees'
        }
        print(f"  ✓ Test Accuracy: {xgb_test_acc:.2f}% | Time: {xgb_train_time:.3f}s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['xgboost'] = None

    # === LightGBM (2017) ===
    print(f"[4/4] Training LightGBM (2017)...")
    try:
        start_time = time.time()
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=attributes, free_raw_data=False)

        params = {
            'objective': 'binary',
            'metric': 'binary_error',
            'num_leaves': 31,
            'learning_rate': 0.3,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        lgb_model = lgb.train(params, train_data, num_boost_round=100)
        lgb_train_time = time.time() - start_time

        # Evaluate
        train_preds = (lgb_model.predict(X_train) > 0.5).astype(int)
        test_preds = (lgb_model.predict(X_test) > 0.5).astype(int)

        lgb_train_acc = 100.0 * np.mean(train_preds == y_train)
        lgb_test_acc = 100.0 * np.mean(test_preds == y_test)
        lgb_overfitting = lgb_train_acc - lgb_test_acc

        results['lightgbm'] = {
            'train_acc': lgb_train_acc,
            'test_acc': lgb_test_acc,
            'overfitting': lgb_overfitting,
            'train_time': lgb_train_time,
            'complexity': '100 trees'
        }
        print(f"  ✓ Test Accuracy: {lgb_test_acc:.2f}% | Time: {lgb_train_time:.3f}s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['lightgbm'] = None

    # Print comparison table
    print(f"\n{'Algorithm':<20} {'Test Acc':<12} {'Train Time':<12} {'Complexity':<15}")
    print("-" * 80)
    for algo_name, algo_key in [('ID3 (1986)', 'id3'), ('C4.5 (1993)', 'c45'),
                                  ('XGBoost (2014)', 'xgboost'), ('LightGBM (2017)', 'lightgbm')]:
        if results[algo_key]:
            r = results[algo_key]
            print(f"{algo_name:<20} {r['test_acc']:>10.2f}% {r['train_time']:>10.3f}s {r['complexity']:<15}")

    # Winner analysis
    print(f"\n{'-'*34}Analysis{'-'*34}")

    test_accs = {k: v['test_acc'] for k, v in results.items() if v}
    best_algo = max(test_accs, key=test_accs.get)
    best_acc = test_accs[best_algo]

    print(f"\n🏆 Best Test Accuracy: {best_algo.upper()} ({best_acc:.2f}%)")

    if results['id3']:
        print(f"\nImprovements over ID3:")
        id3_acc = results['id3']['test_acc']
        for algo_key, display_name in [('c45', 'C4.5'), ('xgboost', 'XGBoost'), ('lightgbm', 'LightGBM')]:
            if results[algo_key]:
                improvement = results[algo_key]['test_acc'] - id3_acc
                print(f"  {display_name:<10} {improvement:+.2f}%")

    print(f"\nOverfitting (Train - Test):")
    for algo_key, display_name in [('id3', 'ID3'), ('c45', 'C4.5'),
                                     ('xgboost', 'XGBoost'), ('lightgbm', 'LightGBM')]:
        if results[algo_key]:
            print(f"  {display_name:<10} {results[algo_key]['overfitting']:.2f}%")

    print(f"\nTraining Speed:")
    if results['id3']:
        id3_time = results['id3']['train_time']
        for algo_key, display_name in [('id3', 'ID3'), ('c45', 'C4.5'),
                                         ('xgboost', 'XGBoost'), ('lightgbm', 'LightGBM')]:
            if results[algo_key]:
                ratio = results[algo_key]['train_time'] / id3_time
                print(f"  {display_name:<10} {results[algo_key]['train_time']:.3f}s ({ratio:.1f}x)")

    return results


def generate_modern_comparison_plot(all_results):
    """Generate 4-panel comparison plot."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('40 Years of Decision Tree Evolution (1986-2017)',
                 fontsize=18, fontweight='bold', y=0.98)

    algorithms = ['ID3\n(1986)', 'C4.5\n(1993)', 'XGBoost\n(2014)', 'LightGBM\n(2017)']
    colors = ['#2196F3', '#9C27B0', '#4CAF50', '#FF9800']
    datasets = list(all_results.keys())

    # Panel 1: Test Accuracy
    ax = axes[0, 0]
    x = np.arange(len(datasets))
    width = 0.2

    for i, (algo, color) in enumerate(zip(['id3', 'c45', 'xgboost', 'lightgbm'], colors)):
        accs = [all_results[ds][algo]['test_acc'] if all_results[ds][algo] else 0
                for ds in datasets]
        ax.bar(x + i*width, accs, width, label=algorithms[i], color=color, alpha=0.8)

    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Test Accuracy Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])

    # Panel 2: Overfitting Control
    ax = axes[0, 1]

    for i, (algo, color) in enumerate(zip(['id3', 'c45', 'xgboost', 'lightgbm'], colors)):
        overfits = [all_results[ds][algo]['overfitting'] if all_results[ds][algo] else 0
                    for ds in datasets]
        ax.bar(x + i*width, overfits, width, label=algorithms[i], color=color, alpha=0.8)

    ax.set_ylabel('Overfitting Gap (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overfitting (Train - Test Accuracy)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Panel 3: Training Speed (log scale)
    ax = axes[1, 0]

    for i, (algo, color) in enumerate(zip(['id3', 'c45', 'xgboost', 'lightgbm'], colors)):
        times = [all_results[ds][algo]['train_time'] if all_results[ds][algo] else 0.001
                 for ds in datasets]
        ax.bar(x + i*width, times, width, label=algorithms[i], color=color, alpha=0.8)

    ax.set_ylabel('Training Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Computational Cost', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Panel 4: Average Performance Summary
    ax = axes[1, 1]

    # Calculate averages
    avg_acc = []
    avg_overfit = []

    for algo in ['id3', 'c45', 'xgboost', 'lightgbm']:
        accs = [all_results[ds][algo]['test_acc'] for ds in datasets if all_results[ds][algo]]
        overfits = [all_results[ds][algo]['overfitting'] for ds in datasets if all_results[ds][algo]]
        avg_acc.append(np.mean(accs) if accs else 0)
        avg_overfit.append(np.mean(overfits) if overfits else 0)

    x_pos = np.arange(len(algorithms))
    ax2 = ax.twinx()

    bars1 = ax.bar(x_pos - 0.2, avg_acc, 0.4, label='Avg Test Accuracy', color='#4CAF50', alpha=0.8)
    bars2 = ax2.bar(x_pos + 0.2, avg_overfit, 0.4, label='Avg Overfitting', color='#F44336', alpha=0.8)

    ax.set_ylabel('Average Test Accuracy (%)', fontsize=12, fontweight='bold', color='#4CAF50')
    ax2.set_ylabel('Average Overfitting (%)', fontsize=12, fontweight='bold', color='#F44336')
    ax.set_title('Average Performance Summary', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, fontsize=10)
    ax.tick_params(axis='y', labelcolor='#4CAF50')
    ax2.tick_params(axis='y', labelcolor='#F44336')
    ax.grid(axis='y', alpha=0.3)

    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('outputs/modern_comparison_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Saved outputs/modern_comparison_plot.png")
    plt.close()


def main():
    """Run comprehensive comparison."""

    print("="*80)
    print("MODERN ML EVOLUTION: 1986-2017")
    print("="*80)
    print("\nComparing four generations of decision tree algorithms:")
    print("  • ID3 (1986): Original information gain")
    print("  • C4.5 (1993): Gain ratio + pruning")
    print("  • XGBoost (2014): Gradient boosting revolution")
    print("  • LightGBM (2017): Modern state-of-the-art")
    print()

    # Load datasets
    print("[1/3] Loading datasets...")
    mushroom = download_mushroom_dataset()
    tictactoe = download_tic_tac_toe_dataset()
    voting = download_voting_dataset()

    print(f"  ✓ Mushroom: {len(mushroom)} examples")
    print(f"  ✓ Tic-Tac-Toe: {len(tictactoe)} examples")
    print(f"  ✓ Voting: {len(voting)} examples")

    # Run comparisons
    print("\n[2/3] Running comparisons...")
    all_results = {}

    all_results['Mushroom'] = compare_all_algorithms('Mushroom Classification', mushroom, 'class')
    all_results['Tic-Tac-Toe'] = compare_all_algorithms('Tic-Tac-Toe Endgame', tictactoe, 'class')
    all_results['Voting'] = compare_all_algorithms('Congressional Voting', voting, 'class')

    # Generate visualization
    print("\n[3/3] Generating visualization...")
    generate_modern_comparison_plot(all_results)

    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY: 40 YEARS OF PROGRESS")
    print("="*80)

    # Calculate grand averages
    grand_avgs = {}
    for algo in ['id3', 'c45', 'xgboost', 'lightgbm']:
        accs = []
        overfits = []
        times = []
        for ds in all_results:
            if all_results[ds][algo]:
                accs.append(all_results[ds][algo]['test_acc'])
                overfits.append(all_results[ds][algo]['overfitting'])
                times.append(all_results[ds][algo]['train_time'])

        if accs:
            grand_avgs[algo] = {
                'avg_acc': np.mean(accs),
                'avg_overfit': np.mean(overfits),
                'avg_time': np.mean(times)
            }

    print(f"\n{'Algorithm':<20} {'Avg Accuracy':<15} {'Avg Overfitting':<18} {'Avg Time':<12}")
    print("-" * 80)
    for algo, name in [('id3', 'ID3 (1986)'), ('c45', 'C4.5 (1993)'),
                       ('xgboost', 'XGBoost (2014)'), ('lightgbm', 'LightGBM (2017)')]:
        if algo in grand_avgs:
            g = grand_avgs[algo]
            print(f"{name:<20} {g['avg_acc']:>13.2f}% {g['avg_overfit']:>16.2f}% {g['avg_time']:>10.3f}s")

    print("\n" + "-"*32 + "Key Insights" + "-"*32)

    if 'id3' in grand_avgs and 'lightgbm' in grand_avgs:
        id3_acc = grand_avgs['id3']['avg_acc']
        lgb_acc = grand_avgs['lightgbm']['avg_acc']
        improvement = lgb_acc - id3_acc

        id3_overfit = grand_avgs['id3']['avg_overfit']
        lgb_overfit = grand_avgs['lightgbm']['avg_overfit']
        overfit_reduction = ((id3_overfit - lgb_overfit) / id3_overfit) * 100

        id3_time = grand_avgs['id3']['avg_time']
        lgb_time = grand_avgs['lightgbm']['avg_time']
        time_cost = lgb_time / id3_time

        print(f"\n1. ACCURACY GAIN: LightGBM achieves {improvement:+.2f}% better accuracy than ID3")
        print(f"   - 31 years of algorithm evolution delivers measurable improvements")

        print(f"\n2. OVERFITTING CONTROL: {overfit_reduction:.1f}% reduction in overfitting")
        print(f"   - ID3: {id3_overfit:.2f}% overfitting")
        print(f"   - LightGBM: {lgb_overfit:.2f}% overfitting")

        print(f"\n3. COMPUTATIONAL COST: {time_cost:.1f}x slower than ID3")
        print(f"   - But modern gradient boosting is highly optimized")
        print(f"   - LightGBM is faster than XGBoost on large datasets")

        print(f"\n4. STATE-OF-THE-ART (2017-present):")
        print(f"   - LightGBM: Faster training, better memory efficiency")
        print(f"   - Histogram-based algorithms for speed")
        print(f"   - GOSS (Gradient-based One-Side Sampling)")
        print(f"   - EFB (Exclusive Feature Bundling)")

        print(f"\n5. WHEN TO USE WHAT:")
        print(f"   - ID3/C4.5: Education, interpretability")
        print(f"   - XGBoost: Balanced performance, widely tested")
        print(f"   - LightGBM: Large datasets, speed critical, state-of-the-art")

    print("\n" + "="*80)
    print("Comparison complete! Files generated:")
    print("  - modern_comparison_plot.png: 4-panel visualization")
    print("="*80)


if __name__ == '__main__':
    main()
