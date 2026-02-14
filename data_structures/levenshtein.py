def min_distance(word1:str, word2:str):
    m,n = len(word1), len(word2)
    
    table = [[0] * (n+1)] * (m+1)
    for i in range(m+1): table[i][0] = i
    for j in range(n+1): table[0][j] = j

    for i in range(1,m+1):
        for j in range(1,n+1):
            
            if word1[i-1] == word2[j-1]:
                table[i][j] = table[i-1][j-1]
            
            else:
                table[i][j] = 1 + min(
                    table[i][j-1], # insert
                    table[i-1][j], # delete
                    table[i-1][j-1], # replace
                )
    
    return table[m][n]


def get_users_viewed(events:list[tuple], p1, p2, T):
    from collections import defaultdict
    user_dict = defaultdict(list)
    for user, time, item in events:
        user_dict[user].append((time, item))

    res = []
    for user, item_list in user_dict.items():
        item_list.sort() # by timestamp -> max complexity O(nlogn)
        last_viewed_time_p1 = None

        for time, item in item_list:
            if item == p1: last_viewed_time_p1 = time
            elif item == p2 and last_viewed_time_p1 is not None:
                if 0 < time - last_viewed_time_p1 <= T:
                    res.append(user)
                    break
    return res


def return_top_k(numbers:list[int], k:int):
    import heapq
    from collections import Counter
    freq = Counter(numbers) # complexity O(n)

    heap = []
    for num, count in freq.items():
        if len(heap) < k:
            heapq.heappush(heap, (count, num))
        else:
            if count > heap[0][0]:
                heapq.heapreplace(heap, (count, num))
    
    return [num for (count,num) in heap]
    

import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    Q = np.matmul(X, W_q)
    K = np.matmul(X, W_k)
    V = np.matmul(X, W_v)
    return Q, K, V


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None):
    # Q: (batch_size, query_len, d_k)
    # K: (batch_size, key_len, d_k)
    # V: (batch_size, value_len, d_v)
    # mask: (batch_size, query_len, key_len)

    dk = Q.shape[-1]
    scaler = np.sqrt(dk)

    scores = np.matmul(Q, K.transpose(0, 2, 1)) / scaler
    if mask is not None:
        scores = scores + mask * -1e9
    attn = softmax(scores, axis=-1)
    output = np.matmul(attn, V)
    return output


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int, mask: np.ndarray = None):
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // n_heads
    
    # Reshape for multi-head: (batch, n_heads, seq_len, d_k)
    Q = Q.reshape(batch_size, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)
    
    # Compute attention for each head
    dk = Q.shape[-1]
    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(dk)
    
    if mask is not None:
        scores = scores + mask[None, None, :, :] * -1e9
    
    attn_weights = softmax(scores, axis=-1)
    attn_out = attn_weights @ V
    
    # Concatenate heads: (batch, seq_len, d_model)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    return attn_out


def layer_normalization(
    X: np.ndarray, 
    gamma: np.ndarray, 
    beta: np.ndarray, 
    epsilon: float = 1e-5
    ) -> np.ndarray:
    """
	Perform Layer Normalization.
    Mean and Var calculated across the last (n-1) dimensions
	"""
    E_x = np.mean(X, axis=-1, keepdims=True)
    Var_x = np.var(X, axis=-1, keepdims=True)
    X = (X - E_x) / np.sqrt(Var_x + epsilon) * gamma + beta
    return X
	