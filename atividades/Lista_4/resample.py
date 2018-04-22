import numpy as np

def split_k_fold(n_elem, n_splits=3, shuffle=True, seed=0):
    idx_train = []
    idx_test  = []
    a = [i for i in range(n_elem)]
    if(shuffle):
        np.random.seed(seed)
        np.random.shuffle(a)
    n_elem_per_fold = int(n_elem / n_splits)
    i = n_elem
    for _ in range(n_splits):
        test = a[i - n_elem_per_fold : i]
        train = a[0 : i-n_elem_per_fold] + a[i:]
        i -= n_elem_per_fold
        idx_train.append(train)
        idx_test.append(test)
    return idx_train, idx_test