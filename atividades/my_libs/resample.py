import numpy as np

def split_train_test(n_elem, perc_train, seed):
    np.random.seed(seed)
    data = [x for x in range(n_elem)]
    np.random.shuffle(data)
    idx_train = data[0: int(n_elem * perc_train)]
    idx_test = data[int(n_elem * perc_train):]
    return idx_train, idx_test

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