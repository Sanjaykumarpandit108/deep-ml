import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5) -> np.ndarray:
    """
    Generate synthetic samples using SMOTE algorithm.

    Note: the random seed is set by the grader before your function runs,
    so you do NOT need to set it. Just use numpy's global RNG directly
    (np.random.randint, np.random.random, ...).

    Args:
        X_minority: 2D array of minority class samples (n_samples, n_features)
        n_synthetic: Number of synthetic samples to generate
        k: Number of nearest neighbors to consider

    Returns:
        2D array of synthetic samples (n_synthetic, n_features)
    """
    n_sample, n_feature = X_minority.shape
    # now calculating the actual number of the neighbours
    k_actual = min(k, n_sample-1) # here we are avoiding the self loop
    # now handling the base case 
    if k_actual == 0 or n_synthetic == 0:
        return np.empty((0, n_feature))

    # now create the empty list
    synthetic_sample = [] # for storing the newly generated sample
    for _ in range(n_synthetic):
        # select the random base sample
        i = np.random.randint(0, n_sample)
        # calcuulate the euclidean distance
        distance = np.sqrt(np.sum((X_minority - X_minority[i]) **2, axis =1))

        # exclude the selected samples
        distance[i] = np.inf
        # now finding the neareast neighbour
        neighbours = np.argsort(distance)[:k_actual]

        # now select one random neighbours
        j = np.random.randint(0,k_actual)
        nn = neighbours[j]

        # now generate the random gap 
        gap = np.random.random()

        # generate the synthetic_sample
        new_sample = X_minority[i] + gap*(X_minority[nn] - X_minority[i])

        synthetic_sample.append(new_sample)

    return np.array(synthetic_sample)



    