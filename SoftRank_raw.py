import torch
import torch.nn as nn
 
class SmoothRank(torch.nn.Module):
 
    def __init__(self, temp=1):
        super(SmoothRank, self).__init__()
        self.temp = temp
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, scores):
        x_0 = scores.unsqueeze(dim=-1)
        x_1 = scores.unsqueeze(dim=-2)
        # print("x_0:", x_0.shape)
        # print("x_1:", x_1.shape)
        diff = x_1 - x_0
        is_lower = self.sigmoid(diff / self.temp)
        ranks = torch.sum(is_lower, dim=-1) + 0.5
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
   
    
class SmoothDCGLoss(nn.Module):
 
    def __init__(self, temp=1):
        super(SmoothDCGLoss, self).__init__()
        self.smooth_ranker = SmoothRank(temp)
        self.zero = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=False)
        self.one = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=False)
 
    def forward(self, scores, labels):
        ranks = self.smooth_ranker(scores)
        d = torch.log2(ranks + 1)
        dg = labels / d
        dcg = dg.sum(dim=-1)
        avg_dcg = dcg.mean()
        loss = -avg_dcg
        return loss
 
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
   
NUM_QUERIES = 3
NUM_DOCS_PER_QUERY = 10
DEVICE = torch.device("cuda:0")
 
# Generate some random scores and labels
scores = torch.rand((NUM_QUERIES, NUM_DOCS_PER_QUERY), dtype=torch.float32, device=DEVICE, requires_grad=False)
# labels = torch.rand((NUM_QUERIES, NUM_DOCS_PER_QUERY), dtype=torch.float32, device=DEVICE, requires_grad=False)
labels = scores
labels = torch.where(labels > 0.8, torch.tensor([1], dtype=torch.float32, device=DEVICE), torch.tensor([0], dtype=torch.float32, device=DEVICE))
 
print_2d_tensor('scores', scores)
print_2d_tensor('labels', labels, prec=0)
 
ranker = SmoothRank(temp=1).to(DEVICE)
ranks = ranker(scores)
print_2d_tensor('ranks (temp=1)', ranks)
 
ranker = SmoothRank(temp=0.001).to(DEVICE)
ranks = ranker(scores)
print_2d_tensor('ranks (temp=0.001)', ranks)
 
criterion = SmoothMRRLoss(temp=1).to(DEVICE)
loss = criterion(scores, labels)
print('smooth mrr loss (temp=1) = ', loss.cpu().numpy().item())
print()
 
criterion = SmoothDCGLoss(temp=1).to(DEVICE)
loss = criterion(scores, labels)
print('smooth dcg loss (temp=1) = ', loss.cpu().numpy().item())
print()