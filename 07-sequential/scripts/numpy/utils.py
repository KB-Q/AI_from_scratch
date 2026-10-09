import numpy as np

def sigmoid(x):
    """σ(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """σ'(x) = σ(x)(1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))"""
    return np.tanh(x)

def tanh_derivative(x):
    """tanh'(x) = 1 - tanh²(x)"""
    return 1 - np.tanh(x)**2

def softmax(x):
    """softmax(x_i) = e^(x_i) / Σ e^(x_j)"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def xavier_init(shape):
    """Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))"""
    return np.random.randn(*shape) * np.sqrt(2.0 / sum(shape))
