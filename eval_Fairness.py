import numpy as np
from eval_metrics import ndcg_list, precision_at_k, recall_at_k, mapk, ndcg_k, idcg_k
from utils import js_divergence
import torch
from SoftRank import SmoothDCGLoss, SmoothRank
import torch.nn.functional as F
import math


def idcg_vector(topk):
    res = torch.zeros(topk)

    for k in range(1, topk + 1):
        res[k - 1] = sum([1.0 / math.log(i + 2, 2) for i in range(k)])

    return res


def Fairness_user(test_list, pred_list, index_F, index_M, user_size, item_size, topk):
    """
    Measure the nDCG@k difference summation between females and males:
    """
    idcg_list = idcg_vector(topk)

    ranks = torch.FloatTensor(np.arange(1, topk + 1)).repeat(user_size, 1)

    labels = None
    for i in range(len(pred_list)):
        item_label = torch.zeros(1, topk)
        for j in range(topk):
            if pred_list[i][j] in test_list[i]:
                item_label[0][j] = 1
        labels = item_label if labels is None else torch.cat((labels, item_label), dim=0)

    d = torch.log2(ranks + 1)
    dg = labels / d

    ndcg = None

    for p in range(1, topk + 1):
        dg_k = dg[:, :p]
        dcg_k = dg_k.sum(dim=-1)
        k = torch.sum(labels, dim=-1).long()
        k = torch.clamp(k, max=p, out=None)
        ndcg_k = (dcg_k / idcg_list[k - 1]).reshape(-1, 1)

        ndcg = ndcg_k if ndcg is None else torch.cat((ndcg, ndcg_k), dim=1)

    ndcg_F = ndcg[index_F]
    ndcg_M = ndcg[index_M]

    # sum the matrix in column
    ndcg_F = ndcg_F.sum(dim=0) / len(index_F)
    ndcg_M = ndcg_M.sum(dim=0) / len(index_M)

    res = torch.abs(torch.log(1 + torch.abs(ndcg_F - ndcg_M))).sum()
    # res = torch.abs(torch.log(ndcg_F+0.5) -+
    # torch.log(ndcg_M+0.5)).sum()

    print("Disparity on user-side:{:.4f}".format(res.item()))

    return res.item()


def Fairness_item(model, genre_mask, target_exposure, pred_list, user_size, topk, args):
    try:
        scores_all = model.myparameters[0].mm(model.myparameters[1].t())
    except:
        scores_all = model.agg_user_embeddings.weight.mm(model.agg_item_embeddings.weight.t())
    scores_top = torch.gather(scores_all.cpu(), 1, torch.LongTensor(pred_list))

    # ranker = SmoothRank(temp=args.temp)
    # ranks = ranker(scores_top, scores_all)
    ranks = torch.FloatTensor(np.arange(1, topk + 1)).repeat(user_size, 1)
    exposure = torch.pow(args.gamma, ranks)

    prob = F.gumbel_softmax(scores_top[:, :topk], tau=1, hard=False)

    sys_exposure = exposure * prob

    genre_top_mask = genre_mask[:, torch.LongTensor(pred_list[:, :topk])].cpu()

    genre_exposure = torch.matmul(genre_top_mask.reshape(genre_mask.shape[0], -1), sys_exposure.reshape(-1, 1))
    genre_exposure = genre_exposure / genre_exposure.sum()

    res = torch.abs(torch.log(1 + torch.abs(genre_exposure - target_exposure.cpu()))).sum()
    # res = torch.abs(torch.log(genre_exposure+0.5) - torch.log(target_exposure.cpu()+0.5)).sum()
    print("Disparity on item-side:{:.4f}".format(res.item()))

    return res.item()


def Fairness_age(test_list, pred_list, age_mask, user_size, topk):
    """
    Measure the nDCG@k difference summation between females and males:
    """
    idcg_list = idcg_vector(topk)

    ranks = torch.FloatTensor(np.arange(1, topk+1)).repeat(user_size, 1)

    labels = None
    for i in range(len(pred_list)):
        item_label = torch.zeros(1, topk)
        for j in range(topk):
            if pred_list[i][j] in test_list[i]:
                item_label[0][j] = 1
        labels = item_label if labels is None else torch.cat((labels, item_label), dim=0)

    d = torch.log2(ranks + 1)
    dg = labels / d

    ndcg = None

    for p in range(1, topk + 1):
        dg_k = dg[:, :p]
        dcg_k = dg_k.sum(dim=-1)
        k = torch.sum(labels, dim=-1).long()
        k = torch.clamp(k, max=p, out=None)
        ndcg_k = (dcg_k / idcg_list[k-1]).reshape(-1, 1)

        ndcg = ndcg_k if ndcg is None else torch.cat((ndcg, ndcg_k), dim=1)


    age_mask = age_mask.cpu()
    ndcg_age = torch.matmul(age_mask, ndcg)
    ndcg_age = ndcg_age / age_mask.sum(dim=1).reshape(7, 1)
    # ndcg_flat = (torch.ones(7, 1) * float(1 / 7)).reshape(-1, 1)
    # ndcg_age = -torch.log(ndcg_age.pow(2))

    res = 0
    for row in range(6):
        for col in range(row + 1, 7):
            res += torch.abs(torch.log(1+torch.abs(ndcg_age[row] - ndcg_age[col]))).sum()
            # res += torch.abs(torch.log(ndcg_age[row]+0.5) - torch.log(ndcg_age[col]+0.5)).sum()

    res = res / 21.0

    print("Disparity on user-side (age):{:.4f}".format(res.item()))

    return res.item()