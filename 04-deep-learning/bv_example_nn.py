"""
Bias-Variance Tradeoff and Double Descent with Neural Networks

This script demonstrates:
1. Classical bias-variance tradeoff with varying network width
2. Double descent phenomenon in deep learning
3. How training epochs affect the double descent curve
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


class SimpleNN(nn.Module):
    """Simple feedforward neural network with configurable width."""
    
    def __init__(self, input_dim=1, hidden_dim=10, n_layers=2):
        super(SimpleNN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_data(n_samples=100, noise_level=0.3, seed=None):
    """Generate synthetic data from a nonlinear function."""
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.uniform(-3, 3, n_samples)
    # True function: combination of sine and polynomial
    y_true = np.sin(X) + 0.2 * X**2 - 0.1 * X**3
    # Add noise
    y = y_true + np.random.normal(0, noise_level, n_samples)
    
    return X.reshape(-1, 1).astype(np.float32), y.astype(np.float32)


def train_model(model, train_loader, epochs=200, lr=0.01, weight_decay=0.0):
    """Train neural network."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    return model


def evaluate_model(model, X, y):
    """Evaluate model and return MSE."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        predictions = model(X_tensor)
        mse = nn.MSELoss()(predictions, y_tensor).item()
    return mse


def experiment_width_vs_error(n_train=50, n_test=200, width_range=None, 
                              n_layers=2, epochs=300, n_trials=10):
    """
    Experiment varying network width to show bias-variance tradeoff.
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        width_range: List of hidden layer widths to test
        n_layers: Number of hidden layers
        epochs: Training epochs
        n_trials: Number of random trials for averaging
    """
    if width_range is None:
        width_range = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    print("=" * 70)
    print("EXPERIMENT 1: Bias-Variance Tradeoff with Network Width")
    print("=" * 70)
    print(f"Training samples: {n_train}, Test samples: {n_test}")
    print(f"Network depth: {n_layers} hidden layers, Epochs: {epochs}")
    print(f"Trials: {n_trials}\n")
    
    # Generate fixed test set
    X_test, y_test = generate_data(n_test, noise_level=0.3)
    
    train_errors = []
    test_errors = []
    param_counts = []
    bias_squared_list = []
    variance_list = []
    
    for width in width_range:
        trial_train_errors = []
        trial_test_errors = []
        test_predictions = []
        
        # Count parameters
        temp_model = SimpleNN(input_dim=1, hidden_dim=width, n_layers=n_layers)
        n_params = count_parameters(temp_model)
        param_counts.append(n_params)
        
        for trial in range(n_trials):
            # Generate new training data each trial
            X_train, y_train = generate_data(n_train, noise_level=0.3, seed=trial)
            
            # Create model
            model = SimpleNN(input_dim=1, hidden_dim=width, n_layers=n_layers)
            
            # Create data loader
            train_dataset = TensorDataset(torch.FloatTensor(X_train), 
                                         torch.FloatTensor(y_train))
            train_loader = DataLoader(train_dataset, batch_size=min(32, n_train), 
                                     shuffle=True)
            
            # Train
            model = train_model(model, train_loader, epochs=epochs, lr=0.001)
            
            # Evaluate
            train_error = evaluate_model(model, X_train, y_train)
            test_error = evaluate_model(model, X_test, y_test)
            
            trial_train_errors.append(train_error)
            trial_test_errors.append(test_error)
            
            # Store predictions for bias-variance decomposition
            model.eval()
            with torch.no_grad():
                preds = model(torch.FloatTensor(X_test)).numpy()
                test_predictions.append(preds)
        
        train_errors.append(np.mean(trial_train_errors))
        test_errors.append(np.mean(trial_test_errors))
        
        # Compute bias and variance
        predictions_array = np.array(test_predictions)
        mean_prediction = predictions_array.mean(axis=0)
        
        bias_squared = np.mean((mean_prediction - y_test) ** 2)
        variance = np.mean(predictions_array.var(axis=0))
        
        bias_squared_list.append(bias_squared)
        variance_list.append(variance)
        
        print(f"Width {width:4d} ({n_params:6d} params): "
              f"Train={train_errors[-1]:.4f}, Test={test_errors[-1]:.4f}, "
              f"Bias²={bias_squared:.4f}, Var={variance:.4f}")
    
    return width_range, param_counts, train_errors, test_errors, bias_squared_list, variance_list


def experiment_double_descent_epochs(n_train=40, n_test=200, width_range=None,
                                    epoch_checkpoints=None, n_layers=2):
    """
    Demonstrate double descent by varying model width and tracking across epochs.
    
    Key insight: Double descent can occur both in model size and training time.
    """
    if width_range is None:
        width_range = list(range(2, 150, 3))
    if epoch_checkpoints is None:
        epoch_checkpoints = [50, 100, 200, 500, 1000]
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Double Descent with Model Width and Training Time")
    print("=" * 70)
    print(f"Training samples: {n_train}, Test samples: {n_test}")
    print(f"Width range: {min(width_range)} to {max(width_range)}")
    print(f"Epoch checkpoints: {epoch_checkpoints}\n")
    
    # Generate fixed datasets
    X_train, y_train = generate_data(n_train, noise_level=0.3)
    X_test, y_test = generate_data(n_test, noise_level=0.3)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), 
                                 torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=min(32, n_train), 
                             shuffle=True)
    
    results = {epochs: {'param_counts': [], 'train_errors': [], 'test_errors': []} 
               for epochs in epoch_checkpoints}
    
    for width in width_range:
        # Create model
        model = SimpleNN(input_dim=1, hidden_dim=width, n_layers=n_layers)
        n_params = count_parameters(model)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Train and checkpoint at different epochs
        model.train()
        epoch_counter = 0
        
        for checkpoint_epoch in epoch_checkpoints:
            epochs_to_train = checkpoint_epoch - epoch_counter
            
            for epoch in range(epochs_to_train):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
            
            epoch_counter = checkpoint_epoch
            
            # Evaluate at checkpoint
            train_error = evaluate_model(model, X_train, y_train)
            test_error = evaluate_model(model, X_test, y_test)
            
            results[checkpoint_epoch]['param_counts'].append(n_params)
            results[checkpoint_epoch]['train_errors'].append(train_error)
            results[checkpoint_epoch]['test_errors'].append(test_error)
        
        # Print progress for key widths and periodically
        if width in [2, 3, 4, 5, 6, 7, 8, 9, 15, 30, 50, 80, 120] or width % 20 == 0:
            print(f"Width {width:3d} ({n_params:5d} params) completed")
    
    print("\nInterpolation threshold (n_train) =", n_train)
    print("Expected peak in test error near this parameter count\n")
    
    return width_range, results, epoch_checkpoints


def plot_width_vs_error(width_range, param_counts, train_errors, test_errors,
                       bias_squared, variance):
    """Visualize bias-variance tradeoff with network width."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Train vs Test Error
    ax = axes[0]
    ax.plot(param_counts, train_errors, 'o-', label='Train Error', 
            linewidth=2, markersize=5)
    ax.plot(param_counts, test_errors, 's-', label='Test Error', 
            linewidth=2, markersize=5)
    min_idx = np.argmin(test_errors)
    ax.axvline(x=param_counts[min_idx], color='red', linestyle='--', 
               alpha=0.5, label=f'Optimal Width ({width_range[min_idx]})')
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Neural Network: Width vs Error', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot 2: Bias-Variance Decomposition
    ax = axes[1]
    total_error = np.array(bias_squared) + np.array(variance)
    ax.plot(param_counts, bias_squared, 'o-', label='Bias²', 
            linewidth=2, markersize=5)
    ax.plot(param_counts, variance, 's-', label='Variance', 
            linewidth=2, markersize=5)
    ax.plot(param_counts, total_error, '^-', label='Bias² + Variance', 
            linewidth=2, markersize=5, color='purple')
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Error Component', fontsize=12)
    ax.set_title('Bias-Variance Decomposition', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('/Users/karthikbalaji/Desktop/databricks/bias_variance/nn_bv.png', 
                dpi=300, bbox_inches='tight')
    print("Plot saved: nn_bv.png")
    plt.show()


def plot_double_descent(width_range, results, epoch_checkpoints, n_train):
    """Visualize double descent phenomenon."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(epoch_checkpoints)))
    
    # Plot 1: Test Error vs Parameters for different epochs
    ax = axes[0]
    for i, epochs in enumerate(epoch_checkpoints):
        param_counts = results[epochs]['param_counts']
        test_errors = results[epochs]['test_errors']
        ax.plot(param_counts, test_errors, '-', label=f'{epochs} epochs', 
                linewidth=2, color=colors[i], alpha=0.8)
    
    # Mark interpolation threshold
    ax.axvline(x=n_train, color='red', linestyle='--', linewidth=2, 
               alpha=0.6, label=f'Interpolation (n={n_train})')
    
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Test Error', fontsize=12)
    ax.set_title('Double Descent: Test Error vs Model Size', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([0.01, 2])
    
    # Plot 2: Train vs Test for final epoch (showing double descent clearly)
    ax = axes[1]
    final_epoch = max(epoch_checkpoints)
    param_counts = results[final_epoch]['param_counts']
    train_errors = results[final_epoch]['train_errors']
    test_errors = results[final_epoch]['test_errors']
    
    ax.plot(param_counts, train_errors, 'o-', label='Train Error', 
            linewidth=2, markersize=3, alpha=0.7)
    ax.plot(param_counts, test_errors, 's-', label='Test Error', 
            linewidth=2, markersize=3, alpha=0.7)
    
    # Annotate regions
    ax.axvspan(min(param_counts), n_train * 0.8, alpha=0.15, color='blue')
    ax.text(n_train * 0.3, ax.get_ylim()[1] * 0.5, 'Under-\nparameterized', 
            ha='center', fontsize=10, fontweight='bold')
    
    ax.axvspan(n_train * 0.8, n_train * 1.5, alpha=0.15, color='red')
    ax.text(n_train, ax.get_ylim()[1] * 0.5, 'Critical\nRegime', 
            ha='center', fontsize=10, fontweight='bold')
    
    ax.axvspan(n_train * 1.5, max(param_counts), alpha=0.15, color='green')
    ax.text(n_train * 3, ax.get_ylim()[1] * 0.5, 'Over-\nparameterized', 
            ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title(f'Double Descent Phases ({final_epoch} epochs)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([0.01, 2])
    
    plt.tight_layout()
    plt.savefig('/Users/karthikbalaji/Desktop/databricks/bias_variance/nn_dd.png', 
                dpi=300, bbox_inches='tight')
    print("Plot saved: nn_dd.png")
    plt.show()


def visualize_nn_predictions(n_train=50, widths=[4, 16, 64, 256], 
                             n_layers=2, epochs=300):
    """Visualize how neural network fits change with model width."""
    X_train, y_train = generate_data(n_train, noise_level=0.3)
    X_test = np.linspace(-3, 3, 200).reshape(-1, 1).astype(np.float32)
    y_true = np.sin(X_test.ravel()) + 0.2 * X_test.ravel()**2 - 0.1 * X_test.ravel()**3
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, width in enumerate(widths):
        ax = axes[idx]
        
        # Train model
        model = SimpleNN(input_dim=1, hidden_dim=width, n_layers=n_layers)
        train_dataset = TensorDataset(torch.FloatTensor(X_train), 
                                     torch.FloatTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=min(32, n_train), 
                                 shuffle=True)
        model = train_model(model, train_loader, epochs=epochs, lr=0.001)
        
        # Predict
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.FloatTensor(X_test)).numpy()
        
        # Plot
        ax.scatter(X_train, y_train, alpha=0.6, s=50, label='Training Data', color='blue')
        ax.plot(X_test, y_true, 'g--', linewidth=2, label='True Function', alpha=0.7)
        ax.plot(X_test, y_pred, 'r-', linewidth=2, label=f'NN (width={width})')
        
        n_params = count_parameters(model)
        train_mse = evaluate_model(model, X_train, y_train)
        ax.set_title(f'Width={width} ({n_params} params)\nTrain MSE: {train_mse:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-4, 4])
    
    plt.tight_layout()
    plt.savefig('/Users/karthikbalaji/Desktop/databricks/bias_variance/nn_fits.png', 
                dpi=300, bbox_inches='tight')
    print("Plot saved: nn_fits.png")
    plt.show()


def main():
    """Run all experiments and create visualizations."""
    print("\n" + "=" * 70)
    print("BIAS-VARIANCE TRADEOFF & DOUBLE DESCENT DEMONSTRATION")
    print("Neural Network Experiments")
    print("=" * 70)
    
    # Experiment 1: Width vs Error (Bias-Variance Tradeoff)
    widths, params, train_err, test_err, bias_sq, variance = \
        experiment_width_vs_error(n_train=50, n_test=200, 
                                 width_range=[2, 4, 8, 16, 32, 64, 128, 256, 512],
                                 n_layers=2, epochs=300, n_trials=5)
    plot_width_vs_error(widths, params, train_err, test_err, bias_sq, variance)
    
    # Experiment 2: Double Descent
    # Create dense width range with extra resolution near critical regime
    # For n_train=40 with 2-layer network, critical width is around 4-6
    width_range_dense = (
        list(range(2, 10, 1)) +          # Dense near critical: 2,3,4,5,6,7,8,9
        list(range(10, 30, 2)) +          # Medium density: 10,12,14,...,28
        list(range(30, 80, 3)) +          # Coarser: 30,33,36,...,78
        list(range(80, 150, 5))           # Sparse in overparameterized: 80,85,...,145
    )
    width_range, results, checkpoints = \
        experiment_double_descent_epochs(n_train=40, n_test=200,
                                        width_range=width_range_dense,
                                        epoch_checkpoints=[50, 100, 200, 500],
                                        n_layers=2)
    plot_double_descent(width_range, results, checkpoints, n_train=40)
    
    # Visualization: Compare fits at different widths
    print("\n" + "=" * 70)
    print("VISUALIZING NEURAL NETWORK FITS AT DIFFERENT WIDTHS")
    print("=" * 70)
    visualize_nn_predictions(n_train=50, widths=[4, 16, 64, 256], epochs=300)
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("1. BIAS-VARIANCE TRADEOFF:")
    print("   - Small networks (few parameters) → High bias, Low variance")
    print("   - Large networks (many parameters) → Low bias, High variance")
    print("   - Optimal network size minimizes test error")
    print()
    print("2. DOUBLE DESCENT IN NEURAL NETWORKS:")
    print("   - Classical regime: Test error decreases with model size")
    print("   - Peak near interpolation threshold (params ≈ samples)")
    print("   - Modern regime: Test error decreases again! (Double descent)")
    print("   - More training can push the curve down and smooth the peak")
    print()
    print("3. PRACTICAL IMPLICATIONS:")
    print("   - Don't fear overparameterization in modern deep learning")
    print("   - Regularization (weight decay) helps in overparameterized regime")
    print("   - Training longer can help escape the critical regime")
    print("   - The 'sweet spot' may be larger than classical theory suggests")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
