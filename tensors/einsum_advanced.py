import numpy as np

def batch_matmul(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Computes Q @ K.T for batched multi-head attention.
    
    Args:
        Q: (B, H, S, D)
        K: (B, H, S, D)

        B: Batch size (number of sequences)
        H: Number of attention heads
        S: Sequence length (number of tokens)
        D: Dimension per head (typically Dmodel/H)
        
    Returns: 
        Scores: Tensor of shape (B, H, S, S)
    """

    """
    The einsum string: "bhid,bhjd->bhij"
        bhid: Query tensor indices (batch, head, query position, dimension)
        bhjd: Key tensor indices (batch, head, key position, dimension)
        bhij: Output indices (batch, head, query position, key position)
    For each (b,h,i,j): Sum over d of Q bhid . K bhjd
    (This is exactly the dot product between query i and key j for each batch and head)
    """

    scores = np.einsum('bhid,bhjd->bhij', Q, K)
    return scores



# Alternative implementation of Q @ K.T for batched multi-head attention using np.matmul.
def batch_matmul_alternative(Q: np.ndarray, K: np.ndarray) -> np.ndarray:

    # Transpose K to (B, H, D, S)
    K_t = np.transpose(K, (0, 1, 3, 2)) 

    # Matrix multiply Q (B, H, S, D) with K_t (B, H, D, S) -> (B, H, S, S)
    scores = np.matmul(Q, K_t)       

    return scores

"""
    For a single batch and head:
        Query: (S,D) matrix
        Key: (S,D) matrix
        Operation: Q K^T -> (S,S) matrix
        Element (i,j): dot product of query i and key j

    Extended to batches and heads:
        Process all (B,H) pairs in parallel
        Result: (B, H, S, S) tensor
"""



# Complete attention 
# Attention = weighted average of values
# Weights come from softmax(QK^T/√d)
def attention(Q, K, V):
    """
    1. Compute scores: QK^T
    2. Scale: / sqrt(d_k)
    3. Softmax: Attention weights
    4. Apply: Weighted sum of values
    """
    d_k = Q.shape[-1]
    
    # Step 1: Our batch_matmul function
    scores = np.einsum("bhid,bhjd->bhij", Q, K)
    
    # Step 2: Scale (prevent softmax saturation)
    scores = scores / np.sqrt(d_k)
    
    # Step 3: Softmax (make weights sum to 1)
    attn_weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    
    # Step 4: Apply to values
    output = np.einsum("bhij,bhjd->bhid", attn_weights, V)
    
    return output, attn_weights