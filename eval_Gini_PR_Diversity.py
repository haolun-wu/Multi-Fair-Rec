import numpy as np
import torch
from SoftRank import SmoothDCGLoss, SmoothRank
import torch.nn.functional as F
import math
import itertools
from collections import Counter
import pandas as pd
# from pygini import gini


# def idcg_vector(topk):
#     res = torch.zeros(topk)
#
#     for k in range(1, topk + 1):
#         res[k - 1] = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
#
#     return res
#
#
# def Fairness_user(test_list, pred_list, index_F, index_M, user_size, item_size, topk):
#     """
#     Measure the nDCG@k difference summation between females and males:
#     """
#     idcg_list = idcg_vector(topk)
#
#     ranks = torch.FloatTensor(np.arange(1, topk + 1)).repeat(user_size, 1)
#
#     labels = None
#     for i in range(len(pred_list)):
#         item_label = torch.zeros(1, topk)
#         for j in range(topk):
#             if pred_list[i][j] in test_list[i]:
#                 item_label[0][j] = 1
#         labels = item_label if labels is None else torch.cat((labels, item_label), dim=0)
#
#     d = torch.log2(ranks + 1)
#     dg = labels / d
#
#     ndcg = None
#
#     for p in range(1, topk + 1):
#         dg_k = dg[:, :p]
#         dcg_k = dg_k.sum(dim=-1)
#         k = torch.sum(labels, dim=-1).long()
#         k = torch.clamp(k, max=p, out=None)
#         ndcg_k = (dcg_k / idcg_list[k - 1]).reshape(-1, 1)
#
#         ndcg = ndcg_k if ndcg is None else torch.cat((ndcg, ndcg_k), dim=1)
#
#     ndcg_F = ndcg[index_F]
#     ndcg_M = ndcg[index_M]
#
#     # sum the matrix in column
#     ndcg_F = ndcg_F.sum(dim=0) / len(index_F)
#     ndcg_M = ndcg_M.sum(dim=0) / len(index_M)
#
#     # max-min ratio
#     max_ndcg = max(ndcg_F[topk-1], ndcg_M[topk-1])
#     min_ndcg = min(ndcg_F[topk-1], ndcg_M[topk-1])
#     print("k={}: max-min ratio:{}".format(topk, max_ndcg / min_ndcg))
#     return
#
#
# # def Fairness_item(model, genre_mask, pred_list, user_size, topk, args):
# #     scores_all = model.myparameters[0].mm(model.myparameters[1].t())
# #     scores_top = torch.gather(scores_all.cpu(), 1, torch.LongTensor(pred_list))
# #
# #     # ranker = SmoothRank(temp=args.temp)
# #     # ranks = ranker(scores_top, scores_all)
# #     ranks = torch.FloatTensor(np.arange(1, topk + 1)).repeat(user_size, 1)
# #     exposure = torch.pow(args.gamma, ranks)
# #
# #     prob = F.gumbel_softmax(scores_top, tau=1, hard=False)
# #
# #     # print("exposure:", exposurhe.shape)
# #     # print("prob:", prob.shape)
# #     sys_exposure = exposure * prob
# #
# #     genre_top_mask = genre_mask[:, torch.LongTensor(pred_list)].cpu()
# #
# #     genre_exposure = torch.matmul(genre_top_mask.reshape(genre_mask.shape[0], -1), sys_exposure.reshape(-1, 1))
# #     genre_exposure = genre_exposure / genre_exposure.sum()
# #
# #     print("genre_exposure:", genre_exposure)
# #
# #     # max-min ratio
# #     max_exp = max(genre_exposure)
# #     min_exp = min(genre_exposure)
# #     print("k={}: {}".format(topk, max_exp / min_exp))
#
# def Fairness_item(occurrence_list):
#     for i in range(len(occurrence_list)):
#         print("k={}: max-min:{}, std:{}".format(5 * (1+i), float(max(occurrence_list[i])) / min(occurrence_list[i]),
#                                                 np.std(np.array(occurrence_list[i]))))
#     return

def Popularity_rate(occurrence_list):
    for i in range(len(occurrence_list)):
        print("k={}: {}".format(5 * (1+i), occurrence_list[i][-1] / sum(occurrence_list[i])))
    return

def Simpson_Diversity(occurrence_list):
    for i in range(len(occurrence_list)):
        nominator = 0
        for val in occurrence_list[i]:
            nominator += val * (val - 1)
        denominator = sum(occurrence_list[i]) * (sum(occurrence_list[i]) - 1)
        print("k={}: {}".format(5 * (1+i), 1 - float(nominator / denominator)))
    return

def gini_coefficient(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.astype("float")
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def Gini(ind_occurrence):
    for i in range(len(ind_occurrence)):
        print("k={}: {}".format(5 * (1+i), gini_coefficient(np.array(ind_occurrence[i]))))
        # print("k={}: {}".format(5 * (1+i), gini(np.array(ind_occurrence[i]).astype("float"))))

    return