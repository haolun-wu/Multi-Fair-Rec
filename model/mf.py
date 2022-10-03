import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable
import torch.nn.functional as F

# if torch.cuda.is_available():
#     import torch.cuda as T
# else:
#     import torch as T


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(MatrixFactorization, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.device = args.device

        self.user_embeddings = nn.Embedding(num_users, args.dim).to(self.device)
        self.item_embeddings = nn.Embedding(num_items, args.dim).to(self.device)

        self.user_embeddings.weight.data = torch.nn.init.normal_(self.user_embeddings.weight.data, 0, 0.01)
        self.item_embeddings.weight.data = torch.nn.init.normal_(self.item_embeddings.weight.data, 0, 0.01)
        
        self.myparameters = [self.user_embeddings.weight, self.item_embeddings.weight]

    def forward(self, user_id, pos_id, neg_id):
        user_emb = self.user_embeddings(user_id)
        pos_emb = self.item_embeddings(pos_id)
        neg_emb = self.item_embeddings(neg_id)


        pos_scores = torch.sum(torch.mul(user_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(user_emb, neg_emb), dim=1)

        tmp = pos_scores - neg_scores

        maxi = nn.LogSigmoid()(tmp)
        bpr_loss = -torch.sum(maxi)

        return bpr_loss

    def predict(self, user_id):
        # user_id = Variable(torch.from_numpy(user_id).long(), requires_grad=False).to(self.device)
        user_emb = self.user_embeddings(user_id)
        pred = user_emb.mm(self.item_embeddings.weight.t())


        return pred


