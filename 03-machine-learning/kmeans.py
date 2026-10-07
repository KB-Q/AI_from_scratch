import numpy as np
np.random.seed(42)

def kmeans(X:np.ndarray, k:int=5, iters:int=100):

    # INITIALIZE
    # X shape = (n,m)
    # centroids shape = (5,m)
    rand_idx = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[rand_idx]

    # LOOP    
    for i in range(iters):
        
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        updated_labels = np.argmin(distances, axis=1)