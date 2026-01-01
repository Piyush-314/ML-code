import numpy as np

def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes matrix product C = AB using 3 nested loops.
    """
    # Get dimensions
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"
    
    # Initialize result matrix with zeros
    C = np.zeros((M, N))
    
    for i in range (M):  # Iterate through rows of A     
        for j in range (N): # Iterate through columns of B
            total=0
            for k in range (K): # dot product loop
                total += A[i,k]*B[k,j]
            C[i,j]=total
    
    return C

def matmul_vectorized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes matrix product C = AB using vectorized operations.
    """
    # The @ operator is the standard Python shorthand for matrix multiplication
    # You can also use np.dot(A, B) or np.matmul(A, B)
    return A @ B
    

