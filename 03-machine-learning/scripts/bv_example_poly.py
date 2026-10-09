"""
Bias-Variance Tradeoff and Double Descent with Polynomial Regression

This script demonstrates:
1. Classical bias-variance tradeoff (U-shaped test error curve)
2. Double descent phenomenon (test error decreases again in overparameterized regime)
3. Visualization of how model complexity affects train/test error
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


def generate_data(n_samples=100, noise_level=0.5):
    """Generate synthetic data from a simple function with noise."""
    X = np.random.uniform(-3, 3, n_samples)
    # True function: a simple polynomial
    y_true = np.sin(X) + 0.3 * X**2
    # Add noise
    y = y_true + np.random.normal(0, noise_level, n_samples)
    return X.reshape(-1, 1), y, y_true


def fit_polynomial_model(X_train, y_train, degree, alpha=0.0):
    """Fit polynomial regression model with optional regularization."""
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X_train)
    
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X_poly, y_train)
    
    return model, poly


def evaluate_model(model, poly, X, y):
    """Evaluate model and return MSE."""
    X_poly = poly.transform(X)
    y_pred = model.predict(X_poly)
    return mean_squared_error(y, y_pred)


def experiment_bias_variance_tradeoff(n_train=30, n_test=100, max_degree=25, 
                                     noise_level=0.5, n_trials=50):
    """
    Experiment to visualize bias-variance tradeoff.
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        max_degree: Maximum polynomial degree to test
        noise_level: Standard deviation of noise
        n_trials: Number of trials for averaging
    """
    print("=" * 70)
    print("EXPERIMENT 1: Classical Bias-Variance Tradeoff")
    print("=" * 70)
    print(f"Training samples: {n_train}, Test samples: {n_test}")
    print(f"Noise level: {noise_level}, Trials for averaging: {n_trials}")
    print()
    
    degrees = range(1, max_degree + 1)
    train_errors = []
    test_errors = []
    bias_squared_list = []
    variance_list = []
    
    # Generate fixed test set for fair comparison
    X_test, y_test, _ = generate_data(n_test, noise_level)
    
    for degree in degrees:
        trial_train_errors = []
        trial_test_errors = []
        test_predictions = []
        
        for trial in range(n_trials):
            # Generate new training data each trial
            X_train, y_train, _ = generate_data(n_train, noise_level)
            
            # Fit model
            model, poly = fit_polynomial_model(X_train, y_train, degree)
            
            # Evaluate
            train_error = evaluate_model(model, poly, X_train, y_train)
            test_error = evaluate_model(model, poly, X_test, y_test)
            
            trial_train_errors.append(train_error)
            trial_test_errors.append(test_error)
            
            # Store predictions for bias-variance decomposition
            X_test_poly = poly.transform(X_test)
            test_predictions.append(model.predict(X_test_poly))
        
        train_errors.append(np.mean(trial_train_errors))
        test_errors.append(np.mean(trial_test_errors))
        
        # Compute bias and variance
        predictions_array = np.array(test_predictions)
        mean_prediction = predictions_array.mean(axis=0)
        
        bias_squared = np.mean((mean_prediction - y_test) ** 2)
        variance = np.mean(predictions_array.var(axis=0))
        
        bias_squared_list.append(bias_squared)
        variance_list.append(variance)
        
        if degree in [1, 3, 5, 10, 15, 20]:
            print(f"Degree {degree:2d}: Train={train_errors[-1]:.4f}, "
                  f"Test={test_errors[-1]:.4f}, Bias²={bias_squared:.4f}, "
                  f"Variance={variance:.4f}")
    
    return degrees, train_errors, test_errors, bias_squared_list, variance_list


def experiment_double_descent(n_train_list=[10, 20, 30, 50], max_degree=60,
                              noise_level=0.3, n_test=100):
    """
    Experiment to demonstrate double descent phenomenon.
    
    The key insight: when model capacity >> number of samples (interpolation regime),
    test error can decrease again despite perfect training fit.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Double Descent Phenomenon")
    print("=" * 70)
    print("Varying training set sizes to observe double descent")
    print()
    
    # Generate fixed test set
    X_test, y_test, _ = generate_data(n_test, noise_level)
    
    results = {}
    degrees = range(1, max_degree + 1)
    
    for n_train in n_train_list:
        print(f"\nTraining with n={n_train} samples:")
        train_errors = []
        test_errors = []
        
        # Generate training data
        X_train, y_train, _ = generate_data(n_train, noise_level)
        
        for degree in degrees:
            try:
                # Use small regularization to stabilize overparameterized regime
                alpha = 1e-6 if degree > n_train else 0.0
                model, poly = fit_polynomial_model(X_train, y_train, degree, alpha)
                
                train_error = evaluate_model(model, poly, X_train, y_train)
                test_error = evaluate_model(model, poly, X_test, y_test)
                
                train_errors.append(train_error)
                test_errors.append(test_error)
                
            except:
                # Handle numerical instability
                train_errors.append(np.nan)
                test_errors.append(np.nan)
        
        results[n_train] = {
            'train_errors': train_errors,
            'test_errors': test_errors
        }
        
        # Print key statistics
        interpolation_threshold = n_train
        min_test_idx = np.nanargmin(test_errors)
        print(f"  Interpolation threshold (n={interpolation_threshold})")
        print(f"  Min test error at degree {min_test_idx + 1}: {test_errors[min_test_idx]:.4f}")
    
    return degrees, results, n_train_list


def plot_bias_variance_tradeoff(degrees, train_errors, test_errors, 
                                bias_squared, variance):
    """Create visualization for bias-variance tradeoff."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Train vs Test Error
    ax = axes[0]
    ax.plot(degrees, train_errors, 'o-', label='Train Error', linewidth=2, markersize=4)
    ax.plot(degrees, test_errors, 's-', label='Test Error', linewidth=2, markersize=4)
    ax.axvline(x=np.argmin(test_errors) + 1, color='red', linestyle='--', 
               alpha=0.5, label='Optimal Complexity')
    ax.set_xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Classical Bias-Variance Tradeoff', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Bias-Variance Decomposition
    ax = axes[1]
    total_error = np.array(bias_squared) + np.array(variance)
    ax.plot(degrees, bias_squared, 'o-', label='Bias²', linewidth=2, markersize=4)
    ax.plot(degrees, variance, 's-', label='Variance', linewidth=2, markersize=4)
    ax.plot(degrees, total_error, '^-', label='Bias² + Variance', 
            linewidth=2, markersize=4, color='purple')
    ax.set_xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
    ax.set_ylabel('Error Component', fontsize=12)
    ax.set_title('Bias-Variance Decomposition', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('/Users/karthikbalaji/Desktop/databricks/bias_variance_tradeoff.png', 
                dpi=300, bbox_inches='tight')
    print("\nPlot saved: bias_variance_tradeoff.png")
    plt.show()


def plot_double_descent(degrees, results, n_train_list):
    """Create visualization for double descent phenomenon."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(n_train_list)))
    
    # Plot 1: Test Error across different sample sizes
    ax = axes[0]
    for i, n_train in enumerate(n_train_list):
        test_errors = results[n_train]['test_errors']
        ax.plot(degrees, test_errors, '-', label=f'n={n_train}', 
                linewidth=2, color=colors[i], alpha=0.8)
        # Mark interpolation threshold
        ax.axvline(x=n_train, color=colors[i], linestyle=':', alpha=0.4)
    
    ax.set_xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
    ax.set_ylabel('Test Error', fontsize=12)
    ax.set_title('Double Descent: Test Error vs Complexity', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, title='Training Size')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim([0.01, 10])
    
    # Plot 2: Zoomed view on a specific sample size showing descent phases
    ax = axes[1]
    n_train = n_train_list[-1]  # Use largest training set
    test_errors = results[n_train]['test_errors']
    train_errors = results[n_train]['train_errors']
    
    ax.plot(degrees, train_errors, 'o-', label='Train Error', 
            linewidth=2, markersize=3, alpha=0.7)
    ax.plot(degrees, test_errors, 's-', label='Test Error', 
            linewidth=2, markersize=3, alpha=0.7)
    
    # Annotate regions
    ax.axvspan(1, n_train * 0.8, alpha=0.1, color='blue', label='Underparameterized')
    ax.axvspan(n_train * 0.8, n_train * 1.2, alpha=0.1, color='red', label='Critical Region')
    ax.axvspan(n_train * 1.2, max(degrees), alpha=0.1, color='green', label='Overparameterized')
    
    ax.set_xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title(f'Double Descent Phases (n={n_train})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim([0.01, 10])
    
    plt.tight_layout()
    plt.savefig('/Users/karthikbalaji/Desktop/databricks/double_descent_polynomial.png', 
                dpi=300, bbox_inches='tight')
    print("Plot saved: double_descent_polynomial.png")
    plt.show()


def visualize_predictions(n_train=30, degrees_to_plot=[2, 5, 10, 20]):
    """Visualize how polynomial fits change with complexity."""
    X_train, y_train, y_true_train = generate_data(n_train, noise_level=0.5)
    X_test = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_true_test = np.sin(X_test.ravel()) + 0.3 * X_test.ravel()**2
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, degree in enumerate(degrees_to_plot):
        ax = axes[idx]
        
        model, poly = fit_polynomial_model(X_train, y_train, degree)
        X_test_poly = poly.transform(X_test)
        y_pred = model.predict(X_test_poly)
        
        # Plot
        ax.scatter(X_train, y_train, alpha=0.6, s=50, label='Training Data', color='blue')
        ax.plot(X_test, y_true_test, 'g--', linewidth=2, label='True Function', alpha=0.7)
        ax.plot(X_test, y_pred, 'r-', linewidth=2, label=f'Degree {degree} Fit')
        
        train_mse = evaluate_model(model, poly, X_train, y_train)
        ax.set_title(f'Polynomial Degree {degree}\nTrain MSE: {train_mse:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-3, 6])
    
    plt.tight_layout()
    plt.savefig('/Users/karthikbalaji/Desktop/databricks/polynomial_fits_comparison.png', 
                dpi=300, bbox_inches='tight')
    print("Plot saved: polynomial_fits_comparison.png")
    plt.show()


def main():
    """Run all experiments and create visualizations."""
    print("\n" + "=" * 70)
    print("BIAS-VARIANCE TRADEOFF & DOUBLE DESCENT DEMONSTRATION")
    print("Polynomial Regression Experiments")
    print("=" * 70)
    
    # Experiment 1: Bias-Variance Tradeoff
    degrees, train_errors, test_errors, bias_sq, variance = \
        experiment_bias_variance_tradeoff(n_train=30, n_test=100, max_degree=25)
    plot_bias_variance_tradeoff(degrees, train_errors, test_errors, bias_sq, variance)
    
    # Experiment 2: Double Descent
    degrees, results, n_train_list = \
        experiment_double_descent(n_train_list=[10, 20, 30, 50], max_degree=60)
    plot_double_descent(degrees, results, n_train_list)
    
    # Visualization: Compare different polynomial fits
    print("\n" + "=" * 70)
    print("VISUALIZING POLYNOMIAL FITS AT DIFFERENT COMPLEXITIES")
    print("=" * 70)
    visualize_predictions(n_train=30, degrees_to_plot=[2, 5, 10, 20])
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("1. BIAS-VARIANCE TRADEOFF (Classical):")
    print("   - Low complexity → High bias (underfitting)")
    print("   - High complexity → High variance (overfitting)")
    print("   - Optimal complexity minimizes test error")
    print()
    print("2. DOUBLE DESCENT:")
    print("   - Test error first decreases (classical regime)")
    print("   - Peaks near interpolation threshold (n ≈ model parameters)")
    print("   - Decreases again in overparameterized regime!")
    print("   - Regularization helps stabilize the overparameterized region")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
