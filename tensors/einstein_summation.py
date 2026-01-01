import numpy as np
from typing import Dict, Union

def einsum_ops(A: np.ndarray, B: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
    """
    Computes basic ops using np.einsum.
    
    Args:
        A: (N, D)
        B: (D, M)
        
    Returns:
        Dict with "transpose", "sum", "row_sum", "col_sum", "matmul"
    """
    
    # 1. Transpose: Swap indices
    # Input: ij, Output: ji
    transpose = np.einsum('ij->ji', A)
    
    # 2. Sum All: Contract (destroy) all indices
    # Input: ij, Output: (nothing)
    # sum_all = np.einsum('ij->', A)
    sum_all = np.sum(A)
    
    # 3. Row Sum: Keep rows (i), destroy columns (j)
    # Input: ij, Output: i
    # row_sum = np.einsum('ij->i', A)
    row_sum = np.sum(A, axis=1)
    
    # 4. Col Sum: Keep columns (j), destroy rows (i)
    # Input: ij, Output: j
    # col_sum = np.einsum('ij->j', A)
    col_sum = np.sum(A, axis=0)
    
    # 5. MatMul: Standard matrix multiplication
    # A is (N, D) -> 'ik'
    # B is (D, M) -> 'kj' (k must match D)
    # Output is (N, M) -> 'ij'
    # The 'k' index appears in input but not output, so it is summed over.
    # matmul = np.einsum('ik,kj->ij', A, B)
    matmul = A @ B
    
    return {
        "transpose": transpose,
        "sum": sum_all,
        "row_sum": row_sum,
        "col_sum": col_sum,
        "matmul": matmul
    }

"""
# B=Batch, H=Heads, T=Time(Seq), D=Dim
# Query: BHTD, Key: BHTD
# We want attention scores: BHTT (Sequence vs Sequence for each Head)
    scores = np.einsum('bhtd, bhkd -> bhtk', Q, K)
"""

