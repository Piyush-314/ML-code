import numpy as np

def broadcast_ops(X: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Computes {(X + b)*w} using broadcasting (efficient vectorization)
    
    Args:
        X: Input matrix of shape (N, D)
        b: Bias vector of shape (D,)
        w: Weight vector of shape (N,)
        
    Returns:
        Resulting matrix of shape (N, D)
    """
    
    # 1. Add bias 
    # X[N,D], b[D,] -> stretches b to match D
    X_bias = X + b  # Broadcasting works automatically for addition

    # 2. Multiply weight
    # Reshape w from (N,) to (N,1) for broadcasting across rows
    w_column = w[:, None]  # Reshape to (N, 1)
    
    # Multiply element-wise: each row of X_bias is multiplied by corresponding weight
    Y = X_bias * w_column  # Broadcasting works here

    return Y


Result = broadcast_ops(X, b, w)
print(Result)


"""
X = [[1, 2], 
     [3, 4]]  # Shape (2, 2)
b = [10, 20]  # Shape (2,)
w = [0.5, 2]  # Shape (2,)

# Step 1: X + b
# [[1+10, 2+20], 
#  [3+10, 4+20]] 
# = [[11, 22], [13, 24]]

# Step 2: Multiply by w
# w needs to be column vector: [[0.5], [2]]
# Row 0: [11, 22] * 0.5 = [5.5, 11]
# Row 1: [13, 24] * 2   = [26, 48]

Result = [[5.5, 11.], 
          [26., 48.]]"""


'''
Why Broadcasting Matters in ML:
In ML, operations on large datasets and high-dimensional tensors are common. Broadcasting allows for efficient computation without the need for explicit replication of data, which saves memory and speeds up calculations. 

1. Neural Networks: When applying weights and biases to layers of neurons, broadcasting enables efficient addition and multiplication across batches of data.
2. Data Normalization: Operations like mean subtraction and scaling can be performed efficiently across datasets.
'''