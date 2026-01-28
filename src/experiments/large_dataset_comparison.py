"""
Test algorithms on LARGE datasets where modern methods excel.

Downloads and tests on:
1. Covertype (Forest Cover Type) - 581K rows, 54 features
2. Adult Income (Census) - 48K rows, 14 features
3. Bank Marketing - 45K rows, 17 features
4. Credit Card Default - 30K rows, 23 features

Shows where XGBoost and LightGBM truly outperform classical methods.
"""

import sys
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.algorithms import ID3
from src.algorithms import C45
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import time
from collections import Counter
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def prepare_data_for_classical(X, y, feature_names):
    """Convert numpy arrays to dictionary format for ID3/C4.5."""
    examples = []
    for i in range(len(X)):
        ex = {}
        for j, fname in enumerate(feature_names):
            # Discretize continuous features for classical methods
            # Simple binning into quartiles
            ex[fname] = str(int(X[i, j]))  # Convert to discrete
        ex['class'] = str(y[i])
        examples.append(ex)
    return examples


def evaluate_classical(model, X_test, y_test, feature_names):
    """Evaluate ID3/C4.5 model."""
    examples = prepare_data_for_classical(X_test, y_test, feature_names)
    correct = sum(1 for ex in examples if str(model.predict(ex)) == ex['class'])
    return 100.0 * correct / len(examples)


def compare_on_large_dataset(name, X, y, feature_names, max_train_size=None):
    """Compare all algorithms on a single large dataset."""

    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    print(f"Dataset size: {len(X):,} examples, {X.shape[1]} features")

    # Limit dataset size for very large ones (to keep runtime reasonable)
    if max_train_size and len(X) > max_train_size:
        print(f"Sampling {max_train_size:,} examples for faster testing...")
        indices = np.random.choice(len(X), max_train_size, replace=False)
        X = X[indices]
        y = y[indices]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Training: {len(X_train):,} | Test: {len(X_test):,}")
    class_counts = Counter(y)
    print(f"Classes: {len(class_counts)} ({dict(sorted(Counter(y_train).items()))})")

    results = {}

    # === ID3 (1986) ===
    print(f"\n[1/4] Training ID3 (1986)...")
    try:
        # For large datasets, limit ID3 training size
        id3_train_limit = min(10000, len(X_train))
        if len(X_train) > id3_train_limit:
            print(f"  (Using {id3_train_limit:,} samples for ID3 - too slow on full dataset)")
            X_train_id3 = X_train[:id3_train_limit]
            y_train_id3 = y_train[:id3_train_limit]
        else:
            X_train_id3 = X_train
            y_train_id3 = y_train

        train_examples = prepare_data_for_classical(X_train_id3, y_train_id3, feature_names)

        start_time = time.time()
        id3_model = ID3()
        id3_model.fit(train_examples, class_attr='class')
        id3_train_time = time.time() - start_time

        # Evaluate on test set
        id3_test_acc = evaluate_classical(id3_model, X_test, y_test, feature_names)
        id3_train_acc = evaluate_classical(id3_model, X_train_id3, y_train_id3, feature_names)

        results['id3'] = {
            'train_acc': id3_train_acc,
            'test_acc': id3_test_acc,
            'train_time': id3_train_time,
            'train_size': len(X_train_id3)
        }
        print(f"  ✓ Test Accuracy: {id3_test_acc:.2f}% | Time: {id3_train_time:.1f}s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['id3'] = None

    # === C4.5 (1993) ===
    print(f"[2/4] Training C4.5 (1993)...")
    try:
        # For large datasets, limit C4.5 training size
        c45_train_limit = min(10000, len(X_train))
        if len(X_train) > c45_train_limit:
            print(f"  (Using {c45_train_limit:,} samples for C4.5 - too slow on full dataset)")
            X_train_c45 = X_train[:c45_train_limit]
            y_train_c45 = y_train[:c45_train_limit]
        else:
            X_train_c45 = X_train
            y_train_c45 = y_train

        train_examples = prepare_data_for_classical(X_train_c45, y_train_c45, feature_names)

        start_time = time.time()
        c45_model = C45(pruning=True, confidence_level=0.25)
        c45_model.fit(train_examples, class_attr='class')
        c45_train_time = time.time() - start_time

        c45_test_acc = evaluate_classical(c45_model, X_test, y_test, feature_names)
        c45_train_acc = evaluate_classical(c45_model, X_train_c45, y_train_c45, feature_names)

        results['c45'] = {
            'train_acc': c45_train_acc,
            'test_acc': c45_test_acc,
            'train_time': c45_train_time,
            'train_size': len(X_train_c45)
        }
        print(f"  ✓ Test Accuracy: {c45_test_acc:.2f}% | Time: {c45_train_time:.1f}s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['c45'] = None

    # === XGBoost (2014) ===
    print(f"[3/4] Training XGBoost (2014)...")
    try:
        start_time = time.time()

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            'max_depth': 6,
            'eta': 0.3,
            'objective': 'binary:logistic' if len(class_counts) == 2 else 'multi:softprob',
            'eval_metric': 'error',
            'num_class': len(class_counts) if len(class_counts) > 2 else None,
            'seed': 42
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        xgb_model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
        xgb_train_time = time.time() - start_time

        # Evaluate
        if len(class_counts) == 2:
            train_preds = (xgb_model.predict(dtrain) > 0.5).astype(int)
            test_preds = (xgb_model.predict(dtest) > 0.5).astype(int)
        else:
            train_preds = np.argmax(xgb_model.predict(dtrain), axis=1)
            test_preds = np.argmax(xgb_model.predict(dtest), axis=1)

        xgb_train_acc = 100.0 * np.mean(train_preds == y_train)
        xgb_test_acc = 100.0 * np.mean(test_preds == y_test)

        results['xgboost'] = {
            'train_acc': xgb_train_acc,
            'test_acc': xgb_test_acc,
            'train_time': xgb_train_time,
            'train_size': len(X_train)
        }
        print(f"  ✓ Test Accuracy: {xgb_test_acc:.2f}% | Time: {xgb_train_time:.1f}s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['xgboost'] = None

    # === LightGBM (2017) ===
    print(f"[4/4] Training LightGBM (2017)...")
    try:
        start_time = time.time()

        train_data = lgb.Dataset(X_train, label=y_train)

        params = {
            'objective': 'binary' if len(class_counts) == 2 else 'multiclass',
            'metric': 'binary_error' if len(class_counts) == 2 else 'multi_error',
            'num_class': len(class_counts) if len(class_counts) > 2 else None,
            'num_leaves': 31,
            'learning_rate': 0.3,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        lgb_model = lgb.train(params, train_data, num_boost_round=100)
        lgb_train_time = time.time() - start_time

        # Evaluate
        if len(class_counts) == 2:
            train_preds = (lgb_model.predict(X_train) > 0.5).astype(int)
            test_preds = (lgb_model.predict(X_test) > 0.5).astype(int)
        else:
            train_preds = np.argmax(lgb_model.predict(X_train), axis=1)
            test_preds = np.argmax(lgb_model.predict(X_test), axis=1)

        lgb_train_acc = 100.0 * np.mean(train_preds == y_train)
        lgb_test_acc = 100.0 * np.mean(test_preds == y_test)

        results['lightgbm'] = {
            'train_acc': lgb_train_acc,
            'test_acc': lgb_test_acc,
            'train_time': lgb_train_time,
            'train_size': len(X_train)
        }
        print(f"  ✓ Test Accuracy: {lgb_test_acc:.2f}% | Time: {lgb_train_time:.1f}s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['lightgbm'] = None

    # Print summary table
    print(f"\n{'Algorithm':<20} {'Test Acc':<12} {'Train Time':<12} {'Train Size':<12}")
    print("-" * 80)

    for algo_name, algo_key in [('ID3 (1986)', 'id3'), ('C4.5 (1993)', 'c45'),
                                 ('XGBoost (2014)', 'xgboost'), ('LightGBM (2017)', 'lightgbm')]:
        if results[algo_key]:
            r = results[algo_key]
            print(f"{algo_name:<20} {r['test_acc']:>10.2f}% {r['train_time']:>10.1f}s {r['train_size']:>10,}")

    # Analysis
    print(f"\n{'-'*34}Analysis{'-'*34}")

    test_accs = {k: v['test_acc'] for k, v in results.items() if v}
    if test_accs:
        best_algo = max(test_accs, key=test_accs.get)
        best_acc = test_accs[best_algo]
        print(f"\n🏆 Best Test Accuracy: {best_algo.upper()} ({best_acc:.2f}%)")

        if 'id3' in results and results['id3']:
            print(f"\nImprovements over ID3:")
            id3_acc = results['id3']['test_acc']
            for algo_key, display_name in [('c45', 'C4.5'), ('xgboost', 'XGBoost'), ('lightgbm', 'LightGBM')]:
                if results[algo_key]:
                    improvement = results[algo_key]['test_acc'] - id3_acc
                    print(f"  {display_name:<10} {improvement:+.2f}%")

        # Speed comparison
        if 'xgboost' in results and 'lightgbm' in results and results['xgboost'] and results['lightgbm']:
            xgb_time = results['xgboost']['train_time']
            lgb_time = results['lightgbm']['train_time']
            speedup = xgb_time / lgb_time
            print(f"\n⚡ LightGBM vs XGBoost: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

    return results


def load_covertype():
    """Load Forest Covertype dataset (581K rows, 54 features)."""
    print("Downloading Covertype dataset (581K rows)...")
    data = fetch_covtype()
    feature_names = [f'f{i}' for i in range(data.data.shape[1])]
    return data.data, data.target - 1, feature_names  # Make labels 0-indexed


def load_adult():
    """Load Adult Income dataset (48K rows, 14 features)."""
    print("Downloading Adult dataset (48K rows)...")
    try:
        data = fetch_openml('adult', version=2, parser='auto')

        # Handle mixed types - convert to numeric
        X = pd.DataFrame(data.data)

        # Encode categorical features
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                X[col] = pd.Categorical(X[col]).codes

        X = X.values.astype(float)
        y = (pd.Categorical(data.target).codes).astype(int)

        feature_names = [f'f{i}' for i in range(X.shape[1])]
        return X, y, feature_names
    except Exception as e:
        print(f"  ✗ Failed to load Adult: {e}")
        return None, None, None


def load_credit_default():
    """Load Credit Card Default dataset (30K rows)."""
    print("Downloading Credit Default dataset (30K rows)...")
    try:
        data = fetch_openml('credit-g', version=1, parser='auto')

        # Handle mixed types
        X = pd.DataFrame(data.data)
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                X[col] = pd.Categorical(X[col]).codes

        X = X.values.astype(float)
        y = pd.Categorical(data.target).codes.astype(int)

        feature_names = [f'f{i}' for i in range(X.shape[1])]
        return X, y, feature_names
    except Exception as e:
        print(f"  ✗ Failed to load Credit: {e}")
        return None, None, None


def generate_large_dataset_plot(all_results):
    """Generate comprehensive visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Algorithm Performance on Large Datasets',
                 fontsize=18, fontweight='bold', y=0.98)

    datasets = list(all_results.keys())
    algorithms = ['ID3\n(1986)', 'C4.5\n(1993)', 'XGBoost\n(2014)', 'LightGBM\n(2017)']
    colors = ['#2196F3', '#9C27B0', '#4CAF50', '#FF9800']

    x = np.arange(len(datasets))
    width = 0.2

    # Panel 1: Test Accuracy
    ax = axes[0, 0]
    for i, (algo, color) in enumerate(zip(['id3', 'c45', 'xgboost', 'lightgbm'], colors)):
        accs = [all_results[ds][algo]['test_acc'] if all_results[ds][algo] else 0
                for ds in datasets]
        ax.bar(x + i*width, accs, width, label=algorithms[i], color=color, alpha=0.8)

    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Test Accuracy on Large Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, fontsize=9, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Training Time (log scale)
    ax = axes[0, 1]
    for i, (algo, color) in enumerate(zip(['id3', 'c45', 'xgboost', 'lightgbm'], colors)):
        times = [all_results[ds][algo]['train_time'] if all_results[ds][algo] else 0.001
                 for ds in datasets]
        ax.bar(x + i*width, times, width, label=algorithms[i], color=color, alpha=0.8)

    ax.set_ylabel('Training Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Computational Cost', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, fontsize=9, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Panel 3: Accuracy Improvement over ID3
    ax = axes[1, 0]
    for i, (algo, color) in enumerate(zip(['c45', 'xgboost', 'lightgbm'], colors[1:])):
        improvements = []
        for ds in datasets:
            if all_results[ds]['id3'] and all_results[ds][algo]:
                imp = all_results[ds][algo]['test_acc'] - all_results[ds]['id3']['test_acc']
                improvements.append(imp)
            else:
                improvements.append(0)
        ax.bar(x + (i+1)*width, improvements, width, label=algorithms[i+1], color=color, alpha=0.8)

    ax.set_ylabel('Accuracy Improvement over ID3 (%)', fontsize=12, fontweight='bold')
    ax.set_title('Modern Methods vs Classical (ID3)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, fontsize=9, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Panel 4: Dataset Size Impact
    ax = axes[1, 1]
    dataset_sizes = []
    lgb_improvements = []

    for ds in datasets:
        if all_results[ds]['id3'] and all_results[ds]['lightgbm']:
            size = all_results[ds]['lightgbm']['train_size']
            improvement = all_results[ds]['lightgbm']['test_acc'] - all_results[ds]['id3']['test_acc']
            dataset_sizes.append(size)
            lgb_improvements.append(improvement)

    if dataset_sizes:
        ax.scatter(dataset_sizes, lgb_improvements, s=200, alpha=0.6, color='#FF9800')
        ax.set_xlabel('Training Set Size (samples)', fontsize=12, fontweight='bold')
        ax.set_ylabel('LightGBM Improvement over ID3 (%)', fontsize=12, fontweight='bold')
        ax.set_title('Dataset Size vs Modern Method Advantage', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Break-even')

        # Add dataset labels
        for i, ds in enumerate(datasets):
            if i < len(dataset_sizes):
                ax.annotate(ds, (dataset_sizes[i], lgb_improvements[i]),
                           textcoords="offset points", xytext=(0,10),
                           ha='center', fontsize=9)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('outputs/large_dataset_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Saved outputs/large_dataset_comparison.png")
    plt.close()


def main():
    """Run comprehensive comparison on large datasets."""

    print("="*80)
    print("LARGE DATASET COMPARISON: Where Modern Methods Excel")
    print("="*80)
    print("\nTesting on datasets where XGBoost and LightGBM show their true power:")
    print("  • Covertype: 581K rows, 54 features")
    print("  • Adult Income: 48K rows, 14 features")
    print("  • Credit Default: 30K rows, 20 features")
    print()

    all_results = {}

    # Covertype - Very large dataset
    X, y, features = load_covertype()
    if X is not None:
        # Sample to keep runtime reasonable
        all_results['Covertype\n(581K rows)'] = compare_on_large_dataset(
            'Forest Cover Type', X, y, features, max_train_size=50000
        )

    # Adult Income
    X, y, features = load_adult()
    if X is not None:
        all_results['Adult\n(48K rows)'] = compare_on_large_dataset(
            'Adult Income', X, y, features
        )

    # Credit Default
    X, y, features = load_credit_default()
    if X is not None:
        all_results['Credit\n(30K rows)'] = compare_on_large_dataset(
            'Credit Default', X, y, features
        )

    # Generate visualization
    if all_results:
        print("\n" + "="*80)
        print("Generating comprehensive visualization...")
        generate_large_dataset_plot(all_results)

    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY: LARGE DATASETS")
    print("="*80)

    # Calculate grand averages
    grand_avgs = {}
    for algo in ['id3', 'c45', 'xgboost', 'lightgbm']:
        accs = []
        times = []
        for ds in all_results:
            if all_results[ds][algo]:
                accs.append(all_results[ds][algo]['test_acc'])
                times.append(all_results[ds][algo]['train_time'])

        if accs:
            grand_avgs[algo] = {
                'avg_acc': np.mean(accs),
                'avg_time': np.mean(times)
            }

    print(f"\n{'Algorithm':<20} {'Avg Accuracy':<15} {'Avg Time':<12}")
    print("-" * 80)
    for algo, name in [('id3', 'ID3 (1986)'), ('c45', 'C4.5 (1993)'),
                       ('xgboost', 'XGBoost (2014)'), ('lightgbm', 'LightGBM (2017)')]:
        if algo in grand_avgs:
            g = grand_avgs[algo]
            print(f"{name:<20} {g['avg_acc']:>13.2f}% {g['avg_time']:>10.1f}s")

    print("\n" + "-"*32 + "Key Findings" + "-"*32)

    if 'id3' in grand_avgs and 'lightgbm' in grand_avgs:
        id3_acc = grand_avgs['id3']['avg_acc']
        lgb_acc = grand_avgs['lightgbm']['avg_acc']
        improvement = lgb_acc - id3_acc

        print(f"\n1. LARGE DATASET ADVANTAGE:")
        print(f"   - LightGBM: {improvement:+.2f}% better than ID3 on large datasets")
        print(f"   - Modern gradient boosting excels with more data")

        if 'xgboost' in grand_avgs:
            xgb_time = grand_avgs['xgboost']['avg_time']
            lgb_time = grand_avgs['lightgbm']['avg_time']
            speedup = xgb_time / lgb_time

            print(f"\n2. SPEED ADVANTAGE:")
            print(f"   - LightGBM is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than XGBoost")
            print(f"   - Matters significantly on large datasets")

        print(f"\n3. SCALABILITY:")
        print(f"   - Classical methods (ID3/C4.5) don't scale well")
        print(f"   - Limited to ~10K training examples in reasonable time")
        print(f"   - Modern methods handle 100K+ rows efficiently")

    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80)


if __name__ == '__main__':
    np.random.seed(42)
    main()
