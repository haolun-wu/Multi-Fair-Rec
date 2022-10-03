import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from math import log2, log1p
# import ot
import torch

def load_rating_data(f_in, num_users, num_items):
    fp = open(f_in)
    lines = fp.readlines()
    X = np.zeros((num_users, num_items))
    for i, line in enumerate(lines):
        strs = line.strip().split('::')[:3]

        # index minus one since the index is started with 1
        user_id = int(strs[0]) - 1
        item_id = int(strs[1]) - 1
        rating = float(strs[2])
        X[user_id, item_id] = rating

    return csr_matrix(X)


def split_data(rating_matrix, test_ratio=0.2, seed=0):
    train_matrix = np.zeros(rating_matrix.shape)
    test_matrix = np.zeros(rating_matrix.shape)

    for user_id, rating_csr in enumerate(rating_matrix):
        items = rating_csr.indices
        rating_array = rating_csr.toarray()[0]
        train_sample, test_sample = train_test_split(items, test_size=test_ratio, random_state=seed)
        train_matrix[user_id][train_sample] = rating_array[train_sample]
        test_matrix[user_id][test_sample] = rating_array[test_sample]
    print("training size:", train_matrix.shape)
    print("testing size:", test_matrix.shape)
    print("training set:", train_matrix[:2])
    print("testing set:", test_matrix[:2])

    return csr_matrix(train_matrix), csr_matrix(test_matrix)

def extract_index(source, condition):
    # pos_f = list(set(source) & set(condition))
    # pos_m = list(set(source) - set(pos_f))
    
    d = {item:idx for idx, item in enumerate(source)}
    inter = list(set(source).intersection(set(condition)))
    pos_f = [d.get(item) for item in inter]

    pos_m = list(set(list(range(len(source)))) - set(pos_f))
    
    return pos_f, pos_m


def kl_divergence(p, q):
    # p = np.array(p).astype(np.float64)
    # q = np.array(q).astype(np.float64)
    p /= p.sum()
    q /= q.sum()

    eps = 1e-6

    p_zero = torch.where(p == 0)[0].numpy()
    q_zero = torch.where(q == 0)[0].numpy()

    res = 0
    if len(p_zero) > 0:
        for i in range(len(p)):
            if i in p_zero:
                p[i] = eps
            else:
                p[i] -= eps / len(p_zero)
    if len(q_zero) > 0:
        for i in range(len(q)):
            if i in q_zero:
                q[i] = eps
            else:
                q[i] -= eps / len(q_zero)

    for i in range(len(p)):
        res += p[i] * torch.log2(p[i] / q[i])

    return res


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)



def cost_matrix(n):
    res = [[0 for _ in range(n)] for _ in range(n)]
    res[0] = [i for i in range(0, n)]
    for i in range(1, n):
        res[i][:i] = list(map(lambda x:x+1, res[i-1][:i])) 
        res[i][i+1:] = list(map(lambda x:x-1, res[i-1][i+1:]))
    return res
    

# def W_distance(p, q):
#     p = np.array(p).astype(np.float64)
#     q = np.array(q).astype(np.float64)
#     p /= p.sum()
#     q /= q.sum()
#     n = len(p)
#     M = cost_matrix(n)
#     return ot.emd2(p, q, M)


# source = [17,46,52,6,12,34,1]
# condition = [52, 46, 34, 23, 44]
# pos_f, pos_m = extract_index(source, condition)
# print("pos_f:", pos_f)
# print("pos_m:", pos_m)
#
# source = torch.tensor(source)
# print(source[pos_f])
# print(source[pos_m])
