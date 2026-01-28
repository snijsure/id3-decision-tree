"""
Generate decision tree visualizations for ID3, C4.5, and XGBoost.
"""

import sys
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.algorithms import ID3
from src.algorithms import C45
import xgboost as xgb
from src.datasets import download_mushroom_dataset, download_tic_tac_toe_dataset, download_voting_dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from collections import Counter

# For XGBoost visualization
try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

def count_nodes(node):
    """Count total nodes in tree."""
    if node is None:
        return 0
    if node.is_leaf():
        return 1
    return 1 + sum(count_nodes(child) for child in node.children.values())

def tree_depth(node):
    """Calculate tree depth."""
    if node is None or node.is_leaf():
        return 0
    if not node.children:
        return 0
    return 1 + max(tree_depth(child) for child in node.children.values())

def visualize_id3_tree(examples, class_attr='class', output_file='id3_tree.png', max_depth=3):
    """Visualize ID3 tree structure."""
    print(f"\n[ID3] Building tree for visualization...")

    # Train ID3
    model = ID3()
    train_size = int(0.7 * len(examples))
    train_set = examples[:train_size]
    model.fit(train_set, class_attr=class_attr)

    print(f"  Tree has {count_nodes(model.root)} nodes, depth {tree_depth(model.root)}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.98, 'ID3 Decision Tree (1986)',
            ha='center', va='top', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.95, f'Information Gain Criterion | {count_nodes(model.root)} nodes | Depth {tree_depth(model.root)}',
            ha='center', va='top', fontsize=10, color='gray')

    # Draw tree recursively
    def draw_node(node, x, y, width, depth=0):
        if node is None:
            return

        if depth > max_depth:
            # Truncate deep trees
            box = FancyBboxPatch((x - 0.04, y - 0.02), 0.08, 0.04,
                                boxstyle="round,pad=0.005",
                                edgecolor='gray', facecolor='lightgray',
                                linewidth=1, linestyle='--')
            ax.add_patch(box)
            ax.text(x, y, '...', ha='center', va='center', fontsize=8, color='gray')
            return

        if node.is_leaf():
            # Leaf node (green)
            box = FancyBboxPatch((x - 0.05, y - 0.025), 0.1, 0.05,
                                boxstyle="round,pad=0.008",
                                edgecolor='darkgreen', facecolor='lightgreen',
                                linewidth=2)
            ax.add_patch(box)
            ax.text(x, y, f"Class:\n{node.label}",
                   ha='center', va='center', fontsize=9, fontweight='bold')
        else:
            # Internal node (blue)
            box = FancyBboxPatch((x - 0.06, y - 0.025), 0.12, 0.05,
                                boxstyle="round,pad=0.008",
                                edgecolor='darkblue', facecolor='lightblue',
                                linewidth=2)
            ax.add_patch(box)

            # Truncate long attribute names
            attr = node.attribute
            if len(attr) > 15:
                attr = attr[:12] + '...'
            ax.text(x, y, attr, ha='center', va='center',
                   fontsize=9, fontweight='bold')

            # Draw children
            children = node.children
            n_children = len(children)
            if n_children > 0 and depth < max_depth:
                child_width = width / n_children
                y_child = y - 0.12

                for i, (value, child) in enumerate(children.items()):
                    x_child = x - width/2 + child_width/2 + i * child_width

                    # Draw edge
                    ax.plot([x, x_child], [y - 0.03, y_child + 0.03],
                           'k-', linewidth=1, alpha=0.5)

                    # Edge label
                    x_mid = (x + x_child) / 2
                    y_mid = (y - 0.03 + y_child + 0.03) / 2
                    label = str(value)
                    if len(label) > 10:
                        label = label[:8] + '...'
                    ax.text(x_mid, y_mid, label,
                           ha='center', va='center', fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white', edgecolor='none', alpha=0.8))

                    # Recursively draw child
                    draw_node(child, x_child, y_child, child_width, depth + 1)

    draw_node(model.root, 0.5, 0.85, 0.9)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', edgecolor='darkblue', label='Decision Node (test attribute)'),
        mpatches.Patch(facecolor='lightgreen', edgecolor='darkgreen', label='Leaf Node (class prediction)')
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved to {output_file}")
    plt.close()


def visualize_c45_tree(examples, class_attr='class', output_file='c45_tree.png', max_depth=3):
    """Visualize C4.5 tree structure."""
    print(f"\n[C4.5] Building tree for visualization...")

    # Train C4.5
    model = C45(pruning=True, confidence_level=0.25)
    train_size = int(0.7 * len(examples))
    train_set = examples[:train_size]
    model.fit(train_set, class_attr=class_attr)

    print(f"  Tree has {count_nodes(model.root)} nodes, depth {tree_depth(model.root)}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.98, 'C4.5 Decision Tree (1993)',
            ha='center', va='top', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.95, f'Gain Ratio + Pruning | {count_nodes(model.root)} nodes | Depth {tree_depth(model.root)}',
            ha='center', va='top', fontsize=10, color='gray')

    # Draw tree recursively
    def draw_node(node, x, y, width, depth=0):
        if node is None:
            return

        if depth > max_depth:
            # Truncate deep trees
            box = FancyBboxPatch((x - 0.04, y - 0.02), 0.08, 0.04,
                                boxstyle="round,pad=0.005",
                                edgecolor='gray', facecolor='lightgray',
                                linewidth=1, linestyle='--')
            ax.add_patch(box)
            ax.text(x, y, '...', ha='center', va='center', fontsize=8, color='gray')
            return

        if node.is_leaf():
            # Leaf node (orange for C4.5)
            box = FancyBboxPatch((x - 0.05, y - 0.025), 0.1, 0.05,
                                boxstyle="round,pad=0.008",
                                edgecolor='darkorange', facecolor='lightyellow',
                                linewidth=2)
            ax.add_patch(box)
            ax.text(x, y, f"Class:\n{node.label}",
                   ha='center', va='center', fontsize=9, fontweight='bold')
        else:
            # Internal node (purple for C4.5)
            box = FancyBboxPatch((x - 0.06, y - 0.025), 0.12, 0.05,
                                boxstyle="round,pad=0.008",
                                edgecolor='purple', facecolor='lavender',
                                linewidth=2)
            ax.add_patch(box)

            # Truncate long attribute names
            attr = node.attribute
            if len(attr) > 15:
                attr = attr[:12] + '...'
            ax.text(x, y, attr, ha='center', va='center',
                   fontsize=9, fontweight='bold')

            # Draw children
            children = node.children
            n_children = len(children)
            if n_children > 0 and depth < max_depth:
                child_width = width / n_children
                y_child = y - 0.12

                for i, (value, child) in enumerate(children.items()):
                    x_child = x - width/2 + child_width/2 + i * child_width

                    # Draw edge
                    ax.plot([x, x_child], [y - 0.03, y_child + 0.03],
                           'k-', linewidth=1, alpha=0.5)

                    # Edge label
                    x_mid = (x + x_child) / 2
                    y_mid = (y - 0.03 + y_child + 0.03) / 2
                    label = str(value)
                    if len(label) > 10:
                        label = label[:8] + '...'
                    ax.text(x_mid, y_mid, label,
                           ha='center', va='center', fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white', edgecolor='none', alpha=0.8))

                    # Recursively draw child
                    draw_node(child, x_child, y_child, child_width, depth + 1)

    draw_node(model.root, 0.5, 0.85, 0.9)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='lavender', edgecolor='purple', label='Decision Node (gain ratio)'),
        mpatches.Patch(facecolor='lightyellow', edgecolor='darkorange', label='Leaf Node (pruned)')
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved to {output_file}")
    plt.close()


def visualize_xgboost_tree(examples, class_attr='class', output_file='xgboost_tree.png'):
    """Visualize one XGBoost tree from the ensemble."""
    print(f"\n[XGBoost] Building ensemble for visualization...")

    # Prepare data
    train_size = int(0.7 * len(examples))
    train_set = examples[:train_size]

    # Get attributes and encode data
    attributes = [k for k in train_set[0].keys() if k != class_attr]
    encodings = {}
    for attr in attributes:
        unique_vals = list(set(ex.get(attr, '?') for ex in train_set))
        encodings[attr] = {val: i for i, val in enumerate(unique_vals)}

    X_train = []
    y_train = []
    class_values = list(set(ex[class_attr] for ex in train_set))
    class_encoding = {val: i for i, val in enumerate(class_values)}

    for ex in train_set:
        features = [encodings[attr].get(ex.get(attr, '?'), 0) for attr in attributes]
        X_train.append(features)
        y_train.append(class_encoding[ex[class_attr]])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=attributes)
    params = {
        'max_depth': 3,  # Shallow for visualization
        'eta': 0.3,
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'seed': 42
    }
    model = xgb.train(params, dtrain, num_boost_round=10)  # Just 10 trees for visualization

    print(f"  Trained ensemble with 10 trees (depth 3)")

    # Visualize the first tree
    if HAS_GRAPHVIZ:
        try:
            # Create graph using graphviz
            graph = xgb.to_graphviz(model, num_trees=0, rankdir='TB')
            graph.format = 'png'
            graph.render(output_file.replace('.png', ''), cleanup=True)
            print(f"  ✓ Saved to {output_file}")
            return
        except Exception as e:
            print(f"  ⚠ Graphviz visualization failed: {e}")
            print(f"  Creating matplotlib-based visualization instead...")
    else:
        print(f"  Creating matplotlib-based visualization (graphviz not available)...")

        # Fallback: create a conceptual diagram
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Title
        ax.text(0.5, 0.95, 'XGBoost Ensemble (2014)',
                ha='center', va='top', fontsize=16, fontweight='bold')
        ax.text(0.5, 0.92, 'Gradient Boosting: 100 Trees Working Together',
                ha='center', va='top', fontsize=10, color='gray')

        # Show ensemble concept
        ax.text(0.5, 0.85, 'Tree₁ + Tree₂ + Tree₃ + ... + Tree₁₀₀ = Final Prediction',
                ha='center', va='center', fontsize=12, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

        # Draw a few representative trees
        tree_y = 0.75
        for i in range(5):
            x = 0.15 + i * 0.15

            # Tree representation
            box = FancyBboxPatch((x - 0.05, tree_y - 0.05), 0.1, 0.1,
                                boxstyle="round,pad=0.008",
                                edgecolor='darkgreen', facecolor='lightgreen',
                                linewidth=2)
            ax.add_patch(box)
            ax.text(x, tree_y, f'Tree\n{i+1}', ha='center', va='center',
                   fontsize=9, fontweight='bold')

        # Ellipsis
        ax.text(0.9, tree_y, '...', ha='center', va='center', fontsize=20)

        # Arrows to combination
        for i in range(5):
            x = 0.15 + i * 0.15
            ax.annotate('', xy=(0.5, 0.55), xytext=(x, tree_y - 0.06),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5))

        # Combination node
        box = FancyBboxPatch((0.35, 0.48), 0.3, 0.12,
                            boxstyle="round,pad=0.01",
                            edgecolor='darkblue', facecolor='lightblue',
                            linewidth=3)
        ax.add_patch(box)
        ax.text(0.5, 0.54, 'Weighted Sum', ha='center', va='center',
               fontsize=11, fontweight='bold')
        ax.text(0.5, 0.50, 'F(x) = Σ ηᵢ·fᵢ(x)', ha='center', va='center',
               fontsize=9, style='italic')

        # Final prediction
        ax.annotate('', xy=(0.5, 0.38), xytext=(0.5, 0.47),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

        box = FancyBboxPatch((0.4, 0.28), 0.2, 0.1,
                            boxstyle="round,pad=0.01",
                            edgecolor='purple', facecolor='plum',
                            linewidth=3)
        ax.add_patch(box)
        ax.text(0.5, 0.33, 'Final\nPrediction', ha='center', va='center',
               fontsize=11, fontweight='bold')

        # Key features
        features_text = """Key XGBoost Features:
• Each tree learns from previous trees' mistakes
• L1/L2 regularization prevents overfitting
• 100 weak learners → 1 strong learner
• Handles missing values automatically"""

        ax.text(0.5, 0.15, features_text, ha='center', va='center',
               fontsize=9, family='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"  ✓ Saved conceptual diagram to {output_file}")
        plt.close()


def main():
    """Generate all tree visualizations."""
    print("="*70)
    print("Generating Decision Tree Visualizations")
    print("="*70)

    # Use Voting dataset (smaller, cleaner trees for visualization)
    print("\nLoading Congressional Voting dataset...")
    examples = download_voting_dataset()
    print(f"  Loaded {len(examples)} examples")

    # Generate visualizations
    visualize_id3_tree(examples, class_attr='class', output_file='id3_tree.png', max_depth=4)
    visualize_c45_tree(examples, class_attr='class', output_file='c45_tree.png', max_depth=4)
    visualize_xgboost_tree(examples, class_attr='class', output_file='xgboost_tree.png')

    print("\n" + "="*70)
    print("Tree visualizations complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - id3_tree.png: ID3 decision tree structure")
    print("  - c45_tree.png: C4.5 decision tree structure")
    print("  - xgboost_tree.png: XGBoost ensemble visualization")


if __name__ == '__main__':
    main()
