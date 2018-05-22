import numpy as np

def split_train_test(n_elem, perc_train, seed):
    np.random.seed(seed)
    data = [x for x in range(n_elem)]
    np.random.shuffle(data)
    idx_train = data[0: int(n_elem * perc_train)]
    idx_test = data[int(n_elem * perc_train):]
    return idx_train, idx_test
