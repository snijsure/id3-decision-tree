"""
ID3 Decision Tree Algorithm
Implementation based on J.R. Quinlan's 1986 paper "Induction of Decision Trees"
"""

import math
from typing import List, Dict, Any, Optional, Set
from collections import Counter


class TreeNode:
    """Represents a node in the decision tree."""

    def __init__(self, attribute: Optional[str] = None, label: Optional[Any] = None):
        self.attribute = attribute  # Attribute to test at this node
        self.label = label  # Class label if this is a leaf
        self.children: Dict[Any, 'TreeNode'] = {}  # Maps attribute values to child nodes

    def is_leaf(self) -> bool:
        return self.label is not None


class ID3:
    """
    ID3 Decision Tree Classifier

    Implements the algorithm from Section 4 of Quinlan's paper using
    information gain as the attribute selection criterion.
    """

    def __init__(self):
        self.root: Optional[TreeNode] = None
        self.attributes: List[str] = []

    def _entropy(self, p: int, n: int) -> float:
        """
        Calculate the information content I(p, n).

        I(p, n) = -p/(p+n) * log2(p/(p+n)) - n/(p+n) * log2(n/(p+n))

        Args:
            p: Number of positive instances
            n: Number of negative instances

        Returns:
            Information in bits
        """
        if p == 0 or n == 0:
            return 0.0

        total = p + n
        p_ratio = p / total
        n_ratio = n / total

        return -(p_ratio * math.log2(p_ratio) + n_ratio * math.log2(n_ratio))

    def _information_gain(self, examples: List[Dict[str, Any]],
                         attribute: str, class_attr: str) -> float:
        """
        Calculate the information gain of an attribute.

        gain(A) = I(p, n) - E(A)

        where E(A) is the expected information after splitting on A.

        Args:
            examples: Training examples
            attribute: Attribute to evaluate
            class_attr: Name of the class attribute

        Returns:
            Information gain in bits
        """
        # Count total positive and negative examples
        class_counts = Counter(ex[class_attr] for ex in examples)
        classes = list(class_counts.keys())

        if len(classes) != 2:
            # For simplicity, this implementation handles binary classification
            # Could be extended to multi-class
            raise ValueError("ID3 implementation currently supports binary classification")

        p = class_counts[classes[0]]
        n = class_counts[classes[1]]

        # Calculate I(p, n)
        total_entropy = self._entropy(p, n)

        # Calculate E(A) - expected information after split
        value_groups = {}
        for ex in examples:
            val = ex[attribute]
            if val not in value_groups:
                value_groups[val] = []
            value_groups[val].append(ex)

        expected_entropy = 0.0
        for val, group in value_groups.items():
            # Count classes in this group
            group_class_counts = Counter(ex[class_attr] for ex in group)
            pi = group_class_counts.get(classes[0], 0)
            ni = group_class_counts.get(classes[1], 0)

            # Weight by proportion of examples
            weight = (pi + ni) / (p + n)
            expected_entropy += weight * self._entropy(pi, ni)

        return total_entropy - expected_entropy

    def _select_best_attribute(self, examples: List[Dict[str, Any]],
                               available_attributes: Set[str],
                               class_attr: str) -> str:
        """
        Select the attribute with maximum information gain.

        Args:
            examples: Training examples
            available_attributes: Attributes that haven't been used yet
            class_attr: Name of the class attribute

        Returns:
            Best attribute name
        """
        best_attr = None
        best_gain = -float('inf')

        for attr in available_attributes:
            gain = self._information_gain(examples, attr, class_attr)
            if gain > best_gain:
                best_gain = gain
                best_attr = attr

        return best_attr

    def _build_tree(self, examples: List[Dict[str, Any]],
                   available_attributes: Set[str],
                   class_attr: str) -> TreeNode:
        """
        Recursively build the decision tree.

        Algorithm from Section 4 of the paper:
        1. If all examples are same class, return leaf with that class
        2. If no attributes left, return leaf with majority class
        3. Otherwise:
           - Select best attribute A
           - Create node for A
           - Partition examples by values of A
           - Recursively build subtrees for each partition

        Args:
            examples: Training examples for this subtree
            available_attributes: Attributes not yet used in path
            class_attr: Name of the class attribute

        Returns:
            Root node of the subtree
        """
        # Count classes in examples
        class_counts = Counter(ex[class_attr] for ex in examples)

        # If empty or all one class, return leaf
        if not examples or len(class_counts) == 1:
            label = class_counts.most_common(1)[0][0] if examples else None
            return TreeNode(label=label)

        # If no attributes left, return leaf with majority class
        if not available_attributes:
            majority_class = class_counts.most_common(1)[0][0]
            return TreeNode(label=majority_class)

        # Select best attribute using information gain
        best_attr = self._select_best_attribute(examples, available_attributes, class_attr)

        # Create node for this attribute
        node = TreeNode(attribute=best_attr)

        # Partition examples by values of best attribute
        value_groups = {}
        for ex in examples:
            val = ex[best_attr]
            if val not in value_groups:
                value_groups[val] = []
            value_groups[val].append(ex)

        # Recursively build subtrees
        remaining_attrs = available_attributes - {best_attr}
        for val, group in value_groups.items():
            child = self._build_tree(group, remaining_attrs, class_attr)
            node.children[val] = child

        return node

    def fit(self, examples: List[Dict[str, Any]], class_attr: str = 'class'):
        """
        Build the decision tree from training examples.

        Args:
            examples: List of training examples (dictionaries)
            class_attr: Name of the attribute containing the class label
        """
        if not examples:
            raise ValueError("Training set cannot be empty")

        # Get all attributes except the class
        self.attributes = [attr for attr in examples[0].keys() if attr != class_attr]

        # Build the tree
        self.root = self._build_tree(examples, set(self.attributes), class_attr)

    def predict(self, example: Dict[str, Any]) -> Any:
        """
        Classify a new example using the decision tree.

        Args:
            example: Dictionary of attribute values

        Returns:
            Predicted class label
        """
        if self.root is None:
            raise ValueError("Model not trained. Call fit() first.")

        node = self.root
        while not node.is_leaf():
            attr = node.attribute
            val = example.get(attr)

            # If value not in tree, return None (could handle better)
            if val not in node.children:
                return None

            node = node.children[val]

        return node.label

    def print_tree(self, node: Optional[TreeNode] = None, indent: str = "",
                   value: str = "") -> None:
        """
        Print the decision tree structure.

        Args:
            node: Current node (uses root if None)
            indent: Current indentation string
            value: Value that led to this node
        """
        if node is None:
            node = self.root

        if node.is_leaf():
            print(f"{indent}└─ {value} → {node.label}")
        else:
            if value:
                print(f"{indent}└─ {value}")
                indent += "    "
            print(f"{indent}{node.attribute}")
            for val, child in node.children.items():
                self.print_tree(child, indent + "  ", str(val))


def load_training_data() -> List[Dict[str, Any]]:
    """
    Load the training set from Table 1 of the paper.

    Returns:
        List of examples with outlook, temperature, humidity, windy, and class
    """
    return [
        {'outlook': 'sunny', 'temperature': 'hot', 'humidity': 'high', 'windy': False, 'class': 'N'},
        {'outlook': 'sunny', 'temperature': 'hot', 'humidity': 'high', 'windy': True, 'class': 'N'},
        {'outlook': 'overcast', 'temperature': 'hot', 'humidity': 'high', 'windy': False, 'class': 'P'},
        {'outlook': 'rain', 'temperature': 'mild', 'humidity': 'high', 'windy': False, 'class': 'P'},
        {'outlook': 'rain', 'temperature': 'cool', 'humidity': 'normal', 'windy': False, 'class': 'P'},
        {'outlook': 'rain', 'temperature': 'cool', 'humidity': 'normal', 'windy': True, 'class': 'N'},
        {'outlook': 'overcast', 'temperature': 'cool', 'humidity': 'normal', 'windy': True, 'class': 'P'},
        {'outlook': 'sunny', 'temperature': 'mild', 'humidity': 'high', 'windy': False, 'class': 'N'},
        {'outlook': 'sunny', 'temperature': 'cool', 'humidity': 'normal', 'windy': False, 'class': 'P'},
        {'outlook': 'rain', 'temperature': 'mild', 'humidity': 'normal', 'windy': False, 'class': 'P'},
        {'outlook': 'sunny', 'temperature': 'mild', 'humidity': 'normal', 'windy': True, 'class': 'P'},
        {'outlook': 'overcast', 'temperature': 'mild', 'humidity': 'high', 'windy': True, 'class': 'P'},
        {'outlook': 'overcast', 'temperature': 'hot', 'humidity': 'normal', 'windy': False, 'class': 'P'},
        {'outlook': 'rain', 'temperature': 'mild', 'humidity': 'high', 'windy': True, 'class': 'N'},
    ]


if __name__ == "__main__":
    # Load the training data from Table 1 of the paper
    training_data = load_training_data()

    print("Training ID3 Decision Tree")
    print("=" * 50)
    print(f"Training set size: {len(training_data)} examples\n")

    # Create and train the model
    model = ID3()
    model.fit(training_data)

    # Print the resulting tree
    print("Learned Decision Tree:")
    print("-" * 50)
    model.print_tree()

    # Test with a new example (from the paper)
    print("\n" + "=" * 50)
    print("Testing Classification")
    print("=" * 50)
    test_example = {
        'outlook': 'overcast',
        'temperature': 'cool',
        'humidity': 'normal',
        'windy': False
    }

    prediction = model.predict(test_example)
    print(f"Example: {test_example}")
    print(f"Prediction: {prediction}")

    # Test on training set to verify correctness
    print("\n" + "=" * 50)
    print("Training Set Accuracy")
    print("=" * 50)
    correct = 0
    for ex in training_data:
        test_ex = {k: v for k, v in ex.items() if k != 'class'}
        pred = model.predict(test_ex)
        if pred == ex['class']:
            correct += 1

    accuracy = correct / len(training_data) * 100
    print(f"Correct: {correct}/{len(training_data)} ({accuracy:.1f}%)")
