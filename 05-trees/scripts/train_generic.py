"""
Examples for the classic tree models: CART, AdaBoost, gradient boosting.

Usage: python3 train.py [cart] [adaboost] [gbm]   (no argument runs all three)
"""

import sys
import numpy as np
from core_cart import CARTClassifier
from core_adaboost import AdaBoostClassifier
from core_gbm import GradientBoostingClassifier


# %%
def example_cart():
    """CART on synthetic 3-class data (Gini vs entropy) and on Iris."""
    from sklearn.datasets import make_classification, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    print("=" * 70)
    print("Decision Tree Example 1: Synthetic Data")
    print("=" * 70)

    X, y = make_classification(n_samples=1000, n_features=10, n_informative=8,
                               n_redundant=2, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nUsing Gini Impurity:")
    tree_gini = CARTClassifier(max_depth=5, criterion='gini')
    tree_gini.fit(X_train, y_train)
    y_pred_gini = tree_gini.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_gini):.4f}")

    print("\nUsing Entropy (Information Gain):")
    tree_entropy = CARTClassifier(max_depth=5, criterion='entropy')
    tree_entropy.fit(X_train, y_train)
    y_pred_entropy = tree_entropy.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_entropy):.4f}")

    print("\n" + "=" * 70)
    print("Decision Tree Example 2: Iris Dataset")
    print("=" * 70)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )

    tree = CARTClassifier(max_depth=3, criterion='gini')
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    print("\n" + "=" * 70)


# %%
def example_adaboost():
    """AdaBoost on synthetic binary data and on Iris."""
    from sklearn.datasets import make_classification, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    print("=" * 70)
    print("AdaBoost Example 1: Binary Classification")
    print("=" * 70)

    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_redundant=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nAdaBoost with Decision Stumps (depth=1):")
    ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    ada.fit(X_train, y_train, verbose=True)
    y_pred = ada.predict(X_test)
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print(f"\nFirst 5 estimator errors: {ada.estimator_errors_[:5]}")
    print(f"Last 5 estimator errors: {ada.estimator_errors_[-5:]}")

    print("\n" + "=" * 70)
    print("AdaBoost Example 2: Multi-class Classification (Iris)")
    print("=" * 70)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )

    ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    ada.fit(X_train, y_train, verbose=False)
    y_pred = ada.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    y_proba = ada.predict_proba(X_test[:5])
    print("\nProbability predictions for first 5 test samples:")
    for i, (true_label, proba) in enumerate(zip(y_test[:5], y_proba)):
        pred_label = np.argmax(proba)
        print(f"Sample {i}: True={iris.target_names[true_label]}, "
              f"Pred={iris.target_names[pred_label]}, "
              f"Proba={proba}")

    print("\n" + "=" * 70)


# %%
def example_gbm():
    """Gradient boosting on synthetic binary data and on Iris."""
    from sklearn.datasets import make_classification, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    print("=" * 70)
    print("Gradient Boosting Example 1: Binary Classification")
    print("=" * 70)

    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_redundant=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining Gradient Boosting Classifier...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8
    )
    gb.fit(X_train, y_train, verbose=True)

    y_pred = gb.predict(X_test)
    y_proba = gb.predict_proba(X_test)

    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\n" + "=" * 70)
    print("Gradient Boosting Example 2: Multi-class Classification (Iris)")
    print("=" * 70)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )

    print("\nTraining Gradient Boosting Classifier...")
    gb = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3
    )
    gb.fit(X_train, y_train, verbose=False)

    y_pred = gb.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    y_proba = gb.predict_proba(X_test[:5])
    print("\nProbability predictions for first 5 test samples:")
    for i, (true_label, proba) in enumerate(zip(y_test[:5], y_proba)):
        pred_label = np.argmax(proba)
        print(f"Sample {i}: True={iris.target_names[true_label]}, "
              f"Pred={iris.target_names[pred_label]}, "
              f"Proba={proba}")

    print("\n" + "=" * 70)


# %%
EXAMPLES = {"cart": example_cart, "adaboost": example_adaboost, "gbm": example_gbm}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(EXAMPLES):
        EXAMPLES[name]()
