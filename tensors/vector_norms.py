import numpy as np

def compute_norms(x):
    # x shape is (N, D)

    """
    compute_norms(x) that takes a matrix x of shape 
    (N,D) representing a batch of N vectors, each of dimension D, and returns a dictionary with two fundamental norms:

    L1 = |x₁| + |x₂| + ... + |xₙ| (sparse solutions)
    L2 = √(x₁² + x₂² + ... + xₙ²) (smooth solutions)
    
    Regularization perspective:
    - L1: Encourages sparse weights (feature selection)
    - L2: Encourages small weights (prevents overfitting)
    """
    
    # L1 Norm 
    # We apply abs(), then sum across columns (axis=1)
    # Theory: Non-differentiable at 0 (subgradient needed)
    l1 = np.sum(np.abs(x), axis=1)
    
    # L2 Norm "l2": The L2 norm (Euclidean norm) for each vector
    # Formula: Sqrt of sum of squares
    # Theory: Differentiable everywhere (except 0)
    # ε prevents division by 0 in normalization
    l2 = np.sqrt(np.sum(x**2, axis=1))
    
    return {
        "l1": l1, 
        "l2": l2
    }

"""
    Loss = Data Loss + λ x Regularization

    With L2: Loss = MSE + λ x ||w||₂²
    Effect: Shrinks all weights proportionally

    With L1: Loss = MSE + λ x ||w||₁
    Effect: Drives some weights exactly to 0


L1 Regularization (Lasso): 
    - Formula: λ∥w∥₁
    - Effect: Pushes weights to exactly zero (sparsity).
    - Use Case: Feature selection.

L2 Regularization (Ridge/Weight Decay): 
    - Formula: λ∥w∥₂²
    - Effect: Penalizes large weights, keeping them small and distributed.
    - Use Case: Prevents overfitting (smooth models).

Gradient Clipping: 
    - Method: If ∥∇L∥₂ > threshold, scale gradient to threshold.
    - Use Case: Prevents exploding gradients in deep networks (RNNs).

Normalization: 
    - Method: x_norm = x / ∥x∥
    - Use Case: Standardizing input scale or weight vectors.

Loss Functions: Many losses incorporate norms, e.g., mean squared error uses L2 norm
"""


# Example usage:
"""

x = [[3, 4],      # First vector: (3, 4)
     [1, -1]]     # Second vector: (1, -1)

# L1 Norm (Manhattan distance):
# Vector 0: |3| + |4| = 3 + 4 = 7
# Vector 1: |1| + |-1| = 1 + 1 = 2
l1_result = [7, 2]

# L2 Norm (Euclidean distance):
# Vector 0: sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5.0
# Vector 1: sqrt(1² + (-1)²) = sqrt(1 + 1) = sqrt(2) ≈ 1.414
l2_result = [5.0, 1.414]
"""