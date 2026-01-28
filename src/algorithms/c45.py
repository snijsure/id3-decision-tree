"""
C4.5 Decision Tree Algorithm
Implementation based on J.R. Quinlan's 1993 book "C4.5: Programs for Machine Learning"

Key improvements over ID3:
1. Gain ratio criterion (reduces bias toward multi-valued attributes)
2. Post-pruning (reduces overfitting)
3. Continuous attribute handling
4. Improved missing value handling
"""

import math
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from collections import Counter
import copy


class C45Node:
    """Represents a node in the C4.5 decision tree."""

    def __init__(self, attribute: Optional[str] = None, label: Optional[Any] = None,
                 threshold: Optional[float] = None):
        self.attribute = attribute  # Attribute to test at this node
        self.label = label  # Class label if this is a leaf
        self.threshold = threshold  # For continuous attributes
        self.children: Dict[Any, 'C45Node'] = {}  # Maps attribute values to child nodes
        self.is_continuous = False  # Whether this tests a continuous attribute
        self.class_distribution: Dict[Any, int] = {}  # For pruning decisions
        self.error_rate = 0.0  # For pruning

    def is_leaf(self) -> bool:
        return self.label is not None


class C45:
    """
    C4.5 Decision Tree Classifier

    Implements Quinlan's C4.5 algorithm with:
    - Gain ratio criterion
    - Pessimistic error pruning
    - Continuous attribute handling
    - Missing value handling
    """

    def __init__(self, min_samples_split: int = 2, pruning: bool = True,
                 confidence_level: float = 0.25):
        """
        Initialize C4.5 classifier.

        Args:
            min_samples_split: Minimum samples required to split a node
            pruning: Whether to perform post-pruning
            confidence_level: Confidence level for pruning (default 0.25)
        """
        self.root: Optional[C45Node] = None
        self.attributes: List[str] = []
        self.continuous_attributes: Set[str] = set()
        self.min_samples_split = min_samples_split
        self.pruning = pruning
        self.confidence_level = confidence_level

    def _entropy(self, class_counts: Dict[Any, int]) -> float:
        """
        Calculate entropy of a class distribution.

        Args:
            class_counts: Dictionary mapping classes to counts

        Returns:
            Entropy in bits
        """
        total = sum(class_counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in class_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def _split_information(self, subsets: List[List[Dict[str, Any]]]) -> float:
        """
        Calculate split information (intrinsic information).

        IV(A) = -sum((|Si|/|S|) * log2(|Si|/|S|))

        Args:
            subsets: List of subsets created by split

        Returns:
            Split information in bits
        """
        total = sum(len(subset) for subset in subsets)
        if total == 0:
            return 0.0

        split_info = 0.0
        for subset in subsets:
            if len(subset) > 0:
                p = len(subset) / total
                split_info -= p * math.log2(p)

        return split_info

    def _information_gain(self, examples: List[Dict[str, Any]],
                         subsets: List[List[Dict[str, Any]]],
                         class_attr: str) -> float:
        """
        Calculate information gain from a split.

        Args:
            examples: All examples before split
            subsets: Subsets after split
            class_attr: Name of class attribute

        Returns:
            Information gain in bits
        """
        # Calculate parent entropy
        parent_classes = Counter(ex[class_attr] for ex in examples)
        parent_entropy = self._entropy(parent_classes)

        # Calculate weighted average of children entropy
        total = len(examples)
        weighted_entropy = 0.0

        for subset in subsets:
            if len(subset) > 0:
                subset_classes = Counter(ex[class_attr] for ex in subset)
                weight = len(subset) / total
                weighted_entropy += weight * self._entropy(subset_classes)

        return parent_entropy - weighted_entropy

    def _gain_ratio(self, examples: List[Dict[str, Any]],
                   subsets: List[List[Dict[str, Any]]],
                   class_attr: str) -> float:
        """
        Calculate gain ratio (normalized information gain).

        GainRatio(A) = Gain(A) / SplitInfo(A)

        This reduces bias toward attributes with many values.

        Args:
            examples: All examples before split
            subsets: Subsets after split
            class_attr: Name of class attribute

        Returns:
            Gain ratio
        """
        gain = self._information_gain(examples, subsets, class_attr)
        split_info = self._split_information(subsets)

        # Avoid division by zero
        if split_info == 0.0:
            return 0.0

        return gain / split_info

    def _find_best_threshold(self, examples: List[Dict[str, Any]],
                            attribute: str, class_attr: str) -> Optional[float]:
        """
        Find the best threshold for splitting a continuous attribute.

        Tests midpoints between consecutive distinct values.

        Args:
            examples: Training examples
            attribute: Continuous attribute name
            class_attr: Name of class attribute

        Returns:
            Best threshold value, or None if no good split found
        """
        # Get sorted unique values
        values = sorted(set(ex[attribute] for ex in examples
                           if ex[attribute] is not None and ex[attribute] != '?'))

        if len(values) < 2:
            return None

        best_threshold = None
        best_gain_ratio = -float('inf')

        # Test midpoints between consecutive values
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2

            # Split examples
            left = [ex for ex in examples if ex[attribute] is not None
                   and ex[attribute] != '?' and ex[attribute] <= threshold]
            right = [ex for ex in examples if ex[attribute] is not None
                    and ex[attribute] != '?' and ex[attribute] > threshold]

            if not left or not right:
                continue

            # Calculate gain ratio
            subsets = [left, right]
            gr = self._gain_ratio(examples, subsets, class_attr)

            if gr > best_gain_ratio:
                best_gain_ratio = gr
                best_threshold = threshold

        return best_threshold

    def _split_discrete(self, examples: List[Dict[str, Any]],
                       attribute: str) -> Dict[Any, List[Dict[str, Any]]]:
        """Split examples on a discrete attribute."""
        splits = {}
        for ex in examples:
            val = ex.get(attribute, '?')
            if val not in splits:
                splits[val] = []
            splits[val].append(ex)
        return splits

    def _split_continuous(self, examples: List[Dict[str, Any]],
                         attribute: str, threshold: float) -> Dict[str, List[Dict[str, Any]]]:
        """Split examples on a continuous attribute using threshold."""
        left = []
        right = []

        for ex in examples:
            val = ex.get(attribute)
            if val is None or val == '?':
                # Handle missing values - could distribute proportionally
                continue
            elif val <= threshold:
                left.append(ex)
            else:
                right.append(ex)

        return {'<= ' + str(threshold): left, '> ' + str(threshold): right}

    def _select_best_attribute(self, examples: List[Dict[str, Any]],
                               available_attributes: Set[str],
                               class_attr: str) -> Tuple[Optional[str], Optional[float], float]:
        """
        Select the attribute with best gain ratio.

        Args:
            examples: Training examples
            available_attributes: Attributes not yet used
            class_attr: Name of class attribute

        Returns:
            (best_attribute, threshold_if_continuous, gain_ratio)
        """
        best_attr = None
        best_threshold = None
        best_gain_ratio = -float('inf')

        # Calculate average gain for filtering
        gains = []
        for attr in available_attributes:
            if attr in self.continuous_attributes:
                threshold = self._find_best_threshold(examples, attr, class_attr)
                if threshold is None:
                    continue
                splits = self._split_continuous(examples, attr, threshold)
                subsets = list(splits.values())
            else:
                splits = self._split_discrete(examples, attr)
                subsets = list(splits.values())

            if len(subsets) > 1:
                gain = self._information_gain(examples, subsets, class_attr)
                gains.append(gain)

        # Use gain ratio only for attributes with at least average gain
        avg_gain = sum(gains) / len(gains) if gains else 0.0

        for attr in available_attributes:
            if attr in self.continuous_attributes:
                threshold = self._find_best_threshold(examples, attr, class_attr)
                if threshold is None:
                    continue
                splits = self._split_continuous(examples, attr, threshold)
                subsets = list(splits.values())
            else:
                splits = self._split_discrete(examples, attr)
                subsets = list(splits.values())

            if len(subsets) < 2:
                continue

            # Check if gain is at least average
            gain = self._information_gain(examples, subsets, class_attr)
            if gain < avg_gain:
                continue

            gr = self._gain_ratio(examples, subsets, class_attr)

            if gr > best_gain_ratio:
                best_gain_ratio = gr
                best_attr = attr
                if attr in self.continuous_attributes:
                    best_threshold = threshold

        return best_attr, best_threshold, best_gain_ratio

    def _build_tree(self, examples: List[Dict[str, Any]],
                   available_attributes: Set[str],
                   class_attr: str) -> C45Node:
        """
        Recursively build the C4.5 decision tree.

        Args:
            examples: Training examples for this subtree
            available_attributes: Attributes not yet used
            class_attr: Name of class attribute

        Returns:
            Root node of the subtree
        """
        # Count classes
        class_counts = Counter(ex[class_attr] for ex in examples)

        # Base cases: empty or all one class
        if not examples or len(class_counts) == 1 or len(examples) < self.min_samples_split:
            label = class_counts.most_common(1)[0][0] if examples else None
            node = C45Node(label=label)
            node.class_distribution = dict(class_counts)
            return node

        # No attributes left
        if not available_attributes:
            majority_class = class_counts.most_common(1)[0][0]
            node = C45Node(label=majority_class)
            node.class_distribution = dict(class_counts)
            return node

        # Select best attribute
        best_attr, threshold, gain_ratio = self._select_best_attribute(
            examples, available_attributes, class_attr)

        if best_attr is None or gain_ratio <= 0:
            majority_class = class_counts.most_common(1)[0][0]
            node = C45Node(label=majority_class)
            node.class_distribution = dict(class_counts)
            return node

        # Create node for this attribute
        node = C45Node(attribute=best_attr, threshold=threshold)
        node.class_distribution = dict(class_counts)

        # Split examples
        if best_attr in self.continuous_attributes:
            node.is_continuous = True
            splits = self._split_continuous(examples, best_attr, threshold)
            remaining_attrs = available_attributes  # Can reuse continuous attrs
        else:
            splits = self._split_discrete(examples, best_attr)
            remaining_attrs = available_attributes - {best_attr}

        # Recursively build subtrees
        for val, subset in splits.items():
            if subset:
                child = self._build_tree(subset, remaining_attrs, class_attr)
                node.children[val] = child
            else:
                # Empty subset - create leaf with majority class of parent
                majority_class = class_counts.most_common(1)[0][0]
                child = C45Node(label=majority_class)
                child.class_distribution = dict(class_counts)
                node.children[val] = child

        return node

    def _prune_tree(self, node: C45Node, examples: List[Dict[str, Any]],
                   class_attr: str) -> C45Node:
        """
        Perform pessimistic error pruning on the tree.

        Args:
            node: Current node
            examples: Examples that reach this node
            class_attr: Name of class attribute

        Returns:
            Pruned node (might be converted to leaf)
        """
        if node.is_leaf():
            return node

        # Recursively prune children first
        for val, child in list(node.children.items()):
            # Get examples for this branch
            if node.is_continuous:
                if '<=' in val:
                    threshold = node.threshold
                    branch_examples = [ex for ex in examples
                                      if ex.get(node.attribute) is not None
                                      and ex.get(node.attribute) != '?'
                                      and ex[node.attribute] <= threshold]
                else:
                    threshold = node.threshold
                    branch_examples = [ex for ex in examples
                                      if ex.get(node.attribute) is not None
                                      and ex.get(node.attribute) != '?'
                                      and ex[node.attribute] > threshold]
            else:
                branch_examples = [ex for ex in examples if ex.get(node.attribute) == val]

            node.children[val] = self._prune_tree(child, branch_examples, class_attr)

        # Calculate error with subtree
        subtree_errors = sum(child.error_rate * len([ex for ex in examples
                                                      if self._classify_example(ex, child, class_attr) != ex[class_attr]])
                            for child in node.children.values())

        # Calculate error if converted to leaf
        if examples:
            majority_class = Counter(ex[class_attr] for ex in examples).most_common(1)[0][0]
            leaf_errors = sum(1 for ex in examples if ex[class_attr] != majority_class)

            # Add pessimistic adjustment
            n = len(examples)
            cf = self.confidence_level
            pessimistic_errors = leaf_errors + 0.5 + cf * math.sqrt(n)

            # Prune if leaf has fewer errors
            if pessimistic_errors <= subtree_errors:
                node.label = majority_class
                node.attribute = None
                node.children = {}
                node.error_rate = pessimistic_errors / n if n > 0 else 0

        return node

    def _classify_example(self, example: Dict[str, Any], node: C45Node,
                         class_attr: str) -> Any:
        """Helper to classify a single example with a given node."""
        if node.is_leaf():
            return node.label

        attr_val = example.get(node.attribute)

        if node.is_continuous:
            if attr_val is None or attr_val == '?':
                # Use majority class from distribution
                return max(node.class_distribution.items(), key=lambda x: x[1])[0]
            elif attr_val <= node.threshold:
                branch = '<= ' + str(node.threshold)
            else:
                branch = '> ' + str(node.threshold)
        else:
            branch = attr_val

        if branch in node.children:
            return self._classify_example(example, node.children[branch], class_attr)
        else:
            # Unknown value - use majority class
            return max(node.class_distribution.items(), key=lambda x: x[1])[0]

    def fit(self, examples: List[Dict[str, Any]], class_attr: str = 'class',
            continuous_attrs: Optional[List[str]] = None):
        """
        Build the C4.5 decision tree from training examples.

        Args:
            examples: List of training examples (dictionaries)
            class_attr: Name of the attribute containing the class label
            continuous_attrs: List of continuous attribute names
        """
        if not examples:
            raise ValueError("Training set cannot be empty")

        # Identify attributes
        self.attributes = [attr for attr in examples[0].keys() if attr != class_attr]

        # Identify continuous attributes
        if continuous_attrs:
            self.continuous_attributes = set(continuous_attrs)
        else:
            # Auto-detect: if all values are numeric, treat as continuous
            self.continuous_attributes = set()
            for attr in self.attributes:
                try:
                    values = [ex[attr] for ex in examples if ex[attr] not in [None, '?']]
                    if values and all(isinstance(v, (int, float)) for v in values):
                        self.continuous_attributes.add(attr)
                except:
                    pass

        # Build tree
        self.root = self._build_tree(examples, set(self.attributes), class_attr)

        # Prune if requested
        if self.pruning:
            self.root = self._prune_tree(self.root, examples, class_attr)

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

        return self._classify_example(example, self.root, 'class')

    def print_tree(self, node: Optional[C45Node] = None, indent: str = "",
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
            dist = node.class_distribution
            print(f"{indent}└─ {value} → {node.label} {dist}")
        else:
            if value:
                print(f"{indent}└─ {value}")
                indent += "    "

            if node.is_continuous:
                print(f"{indent}{node.attribute} (threshold: {node.threshold:.3f})")
            else:
                print(f"{indent}{node.attribute}")

            for val, child in node.children.items():
                self.print_tree(child, indent + "  ", str(val))


if __name__ == "__main__":
    from test_datasets import download_mushroom_dataset, train_test_split, evaluate_model

    print("Testing C4.5 Decision Tree")
    print("=" * 50)

    # Load training data
    training_data = download_mushroom_dataset()

    if training_data:
        print(f"\nTraining on {len(training_data)} examples")

        # Split data
        train_set, test_set = train_test_split(training_data, test_ratio=0.3)

        # Train C4.5
        model = C45(pruning=True)
        model.fit(train_set)

        # Evaluate
        train_acc = evaluate_model(model, train_set)
        test_acc = evaluate_model(model, test_set)

        print(f"\nResults:")
        print(f"  Training accuracy: {train_acc*100:.2f}%")
        print(f"  Test accuracy:     {test_acc*100:.2f}%")

        print("\nDecision Tree (first few levels):")
        model.print_tree()
