import torch
import torch.nn as nn
import math


class SmoothRank(torch.nn.Module):

    def __init__(self, temp=1):
        super(SmoothRank, self).__init__()
        self.temp = temp
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, scores_max_relevant, scores):
        x_0 = scores_max_relevant.unsqueeze(dim=-1)
        x_1 = scores.unsqueeze(dim=-2)
        diff = x_1 - x_0
        is_lower = diff / self.temp
        is_lower = self.sigmoid(is_lower)
        # del diff

        ranks = torch.sum(is_lower, dim=-1) + 0.5
        # del is_lower
        # torch.cuda.empty_cache()
        return ranks


class SmoothMRRLoss(nn.Module):

    def __init__(self, temp=1):
        super(SmoothMRRLoss, self).__init__()
        self.smooth_ranker = SmoothRank(temp)
        self.zero = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=False)
        self.one = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=False)

    def forward(self, scores, labels):
        ranks = self.smooth_ranker(scores)
        labels = torch.where(labels > 0, self.one, self.zero)
        rr = labels / ranks
        rr_max, _ = rr.max(dim=-1)
        mrr = rr_max.mean()
        loss = -mrr
        return loss


# class SmoothDCGLoss(nn.Module):
#
#     def __init__(self, temp=1):
#         super(SmoothDCGLoss, self).__init__()
#         self.smooth_ranker = SmoothRank(temp)
#         self.zero = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=False)
#         self.one = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=False)
#         # self.topk = topk
#
#     def forward(self, scores_top, scores_all, labels):
#         ranks = self.smooth_ranker(scores_top, scores_all)
#         d = torch.log2(ranks + 1)
#         dg = labels / d
#         dcg = dg.sum(dim=-1)
#         # k = torch.sum(labels, dim=1).long()
#         # k = torch.clamp(k, max=self.topk, out=None)
#         # dcg = dcg / self.idcg_vector[k - 1]
#         dcg = dcg
#         # avg_dcg = dcg.mean()
#         # loss = -avg_dcg
#         return dcg


class SmoothDCGLoss(nn.Module):

    def __init__(self, args, topk, temp=1):
        super(SmoothDCGLoss, self).__init__()
        self.smooth_ranker = SmoothRank(temp)
        self.zero = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=False)
        self.one = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=False)
        self.topk = topk
        self.device = args.device
        self.idcg_vector = self.idcg_k()


    def idcg_k(self):
        res = torch.zeros(self.topk).to(self.device)

        for k in range(1, self.topk+1):
            res[k-1] = sum([1.0 / math.log(i+2, 2) for i in range(k)])

        return res

    def forward(self, scores_top, scores, labels):
        ranks = self.smooth_ranker(scores_top, scores)
        # print("ranks:", ranks)
        d = torch.log2(ranks+1)
        dg = labels / d

        ndcg = None

        for p in range(1, self.topk+1):
            dg_k = dg[:,:p]
            dcg_k = dg_k.sum(dim=-1)
            k = torch.sum(labels, dim=-1).long()
            k = torch.clamp(k, max=p, out=None)
            ndcg_k = (dcg_k / self.idcg_vector[k-1]).reshape(-1, 1)

            ndcg = ndcg_k if ndcg is None else torch.cat((ndcg, ndcg_k), dim=1)

        # print("ndcg:", ndcg.shape)

        # dcg = dg.sum(dim=-1)
        # k = torch.sum(labels, dim=-1).long()
        # k = torch.clamp(k, max = self.topk, out=None)
        # dcg = dcg / self.idcg_vector[k-1]
        # dcg = dcg

        return ndcg


def print_2d_tensor(name, value, prec=3):
    print('[{}]'.format(name))
    value = value.cpu().numpy()
    for i in range(len(value)):
        if prec == 0:
            value_i = [int(x) for x in value[i]]
        else:
            value_i = [round(x, prec) for x in value[i]]
        str_i = [str(x) for x in value_i]
        print('q{}: {}'.format(i, ' '.join(str_i)))
    print()

