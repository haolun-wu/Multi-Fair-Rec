import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging
import sys
import time
from argparse import ArgumentParser
from model.mf import MatrixFactorization
from SoftRank import SmoothDCGLoss, SmoothRank
from sampler import NegSampler, negsamp_vectorized_bsearch_preverif
from min_norm_solvers import MinNormSolver, gradient_normalizers
from eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k, idcg_k
from preprocess import generate_rating_matrix, preprocessing
from eval_Fairness import Fairness_user, Fairness_item, Fairness_age
from eval_Gini_PR_Diversity import Gini, Popularity_rate, Simpson_Diversity
import itertools
from collections import Counter, OrderedDict
import pandas as pd
import csv

import warnings
warnings.filterwarnings("ignore")


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="FairMOOP")
    parser.add_argument('--data', type=str, default='ml-100k', choices=['ml-100k', 'ml-1m', 'lastfm'],
                        help="File path for data")
    parser.add_argument('--gpu_id', type=int, default=0)
    # Preprocessing
    parser.add_argument('--val_ratio', type=float, default=0.1, help="Proportion of validation set")
    parser.add_argument('--test_ratio', type=float, default=0.2, help="Proportion of testing set")
    # Model
    parser.add_argument('--dim', type=int, default=64, help="Dimension for embedding")
    parser.add_argument('--gamma', type=float, default=0.75, help="patience factor")
    parser.add_argument('--temp', type=float, default=1e-5, help="temperature. how soft the ranks to be")
    parser.add_argument('--mode', type=str, default='r', help="which loss to use")
    # Optimizer
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="Weight decay factor")
    # Training
    parser.add_argument('--n_epochs', type=int, default=300, help="Number of epoch during training")
    parser.add_argument('--every', type=int, default=300, help="Period for evaluating during training")
    # parser.add_argument('--patience', type=int, default=50, help="patience for early stopping")
    # MOOP
    parser.add_argument('--moop', type=int, default=0, choices=[0, 1])
    parser.add_argument('--type', type=str, default='loss+', choices=['l2', 'loss', 'loss+', 'none'])
    # Hard setting
    parser.add_argument('--scale1', type=float, default=1, help="hard setting scale on obj1")

    return parser.parse_args()

def generate_pred_list(model, train_matrix, topk=20):
    num_users = train_matrix.shape[0]
    batch_size = 1024
    num_batches = int(num_users / batch_size) + 1
    user_indexes = np.arange(num_users)
    pred_list = None

    for batchID in range(num_batches):
        start = batchID * batch_size
        end = start + batch_size

        if batchID == num_batches - 1:
            if start < num_users:
                end = num_users
            else:
                break

        batch_user_index = user_indexes[start:end]
        batch_user_ids = torch.from_numpy(np.array(batch_user_index)).type(torch.LongTensor).to(args.device)

        rating_pred = model.predict(batch_user_ids)
        rating_pred = rating_pred.cpu().data.numpy().copy()
        rating_pred[train_matrix[batch_user_index].toarray() > 0] = 0
        batch_raw_score = rating_pred

        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
        ind = np.argpartition(rating_pred, -topk)
        ind = ind[:, -topk:]
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
        batch_score_list = rating_pred[np.arange(len(rating_pred))[:, None], batch_pred_list]

        if batchID == 0:
            pred_list = batch_pred_list
            score_list = batch_score_list
            raw_score_list = batch_raw_score
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            score_list = np.append(score_list, batch_score_list, axis=0)
            raw_score_list = np.append(raw_score_list, batch_raw_score, axis=0)

    return pred_list, score_list, raw_score_list

def compute_metrics(test_set, pred_list, topk=20):
    precision, recall, MAP, ndcg = [], [], [], []
    for k in [5, 10, 15, 20]:
        precision.append(precision_at_k(test_set, pred_list, k))
        recall.append(recall_at_k(test_set, pred_list, k))
        MAP.append(mapk(test_set, pred_list, k))
        ndcg.append(ndcg_k(test_set, pred_list, k))

    return precision, recall, MAP, ndcg

def neg_item_pre_sampling(train_matrix, num_neg_candidates=500):
    num_users, num_items = train_matrix.shape
    user_neg_items = []
    for user_id in range(num_users):
        pos_items = train_matrix[user_id].indices
        u_neg_item = negsamp_vectorized_bsearch_preverif(pos_items, num_items, num_neg_candidates)
        user_neg_items.append(u_neg_item)
    user_neg_items = np.asarray(user_neg_items)

    return user_neg_items

def statistics_occurrence(top_id, popular_dict):
    pop_occurrence = []
    ind_occurrence = []
    for k in [5, 10, 15, 20]:
        merged_id = list(itertools.chain(*top_id[:, :k]))
        pop_flat = list((pd.Series(merged_id)).map(popular_dict))
        count_genre = sorted(Counter(pop_flat).most_common(), key=lambda tup: tup[0])
        pop_occurrence.append([x[1] for x in count_genre])

        count_ind = Counter(merged_id).most_common()
        ind_occurrence.append([x[1] for x in count_ind])
    return pop_occurrence, ind_occurrence


def train(args):
    # pre-sample a small set of negative samples
    t1 = time.time()
    user_neg_items = neg_item_pre_sampling(train_val_matrix, num_neg_candidates=500)
    pre_samples = {'user_neg_items': user_neg_items}

    print("Pre sampling time:{}".format(time.time() - t1))

    gender_label = np.zeros(len(index_F) + len(index_M))
    for ind in index_F:
        gender_label[ind] = 1

    model = MatrixFactorization(user_size, item_size, args)
    optimizer = torch.optim.Adam(model.myparameters, lr=args.lr, weight_decay=args.weight_decay)

    dcg_loss = SmoothDCGLoss(args=args, topk=50, temp=args.temp)
    ranker = SmoothRank(temp=args.temp)

    sampler = NegSampler(train_val_matrix, pre_samples, batch_size=args.batch_size, num_neg=1, n_workers=4)

    # saved_path = './saved/{}-{}-{}-{}-bprmf.pt'.format(args.data, args.mode, str(args.moop), args.u)
    # saved_record_path = './saved/{}/{}/record-{}-{}-bprmf.txt'.format(args.data, args.u, args.mode, str(args.moop))
    # saved_fair_ui_path = './saved/{}/{}/fairui-{}-{}-bprmf.txt'.format(args.data, args.u, args.mode, str(args.moop))
    #
    # early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=saved_path)

    num_batches = train_val_matrix.count_nonzero() // args.batch_size

    loss = {}
    loss_record = [[], [], []]
    fairness_record = []
    scale = {}
    loss_clone = {}
    loss_data = {}
    grads = {}
    tasks = []
    s = [[], [], []]

    # build tasks
    if 'r' in args.mode:
        tasks.append('1')
    if 'u' in args.mode:
        tasks.append('2')
    if 'i' in args.mode:
        tasks.append('3')

    # Target exposure on movie genres:
    target_exposure = (torch.ones(genre_num, 1) * float(1 / genre_num)).to(args.device)

    # Sample max_pos items for each user
    neg_ids_list = []
    pos_ids_list = []

    for i in range(user_size):
        if (len(train_user_list[i]) > max_pos):
            sampled_pos_ids = np.random.choice(len(train_user_list[i]), size=max_pos, replace=False)
            tmp = [train_user_list[i][j] for j in sampled_pos_ids]
            pos_ids_list.append(tmp)
        else:
            pos_ids_list.append(train_user_list[i])
        neg_ids_list.append(negsamp_vectorized_bsearch_preverif(np.array(train_user_list[i]), item_size,
                                                                n_samp=max_pos - len(pos_ids_list[i])))

    sampled_ids = np.ones((user_size, max_pos)) * item_size
    labels = np.zeros((user_size, max_pos))

    for i in range(user_size):
        sampled_ids[i][:len(pos_ids_list[i])] = np.array(pos_ids_list[i])
        sampled_ids[i][len(pos_ids_list[i]):] = neg_ids_list[i]
        labels[i][:len(pos_ids_list[i])] = 1

    sampled_ids = torch.LongTensor(sampled_ids).to(args.device)
    labels = torch.LongTensor(labels).to(args.device)


    try:
        for iter in range(args.n_epochs):
            print("Epochs:", iter + 1)

            start = time.time()
            model.train()

            # Start Training
            for batch_id in range(num_batches):
                user, pos, neg = sampler.next_batch()
                neg = np.squeeze(neg)
                unique_u = torch.LongTensor(list(set(user.tolist())))

                user_id = torch.from_numpy(user).type(torch.LongTensor).to(args.device)
                pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(args.device)
                neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(args.device)

                if 'r' in args.mode:
                    loss['1'] = model(user_id, pos_id, neg_id)
                else:
                    loss['1'] = torch.tensor(0)

                if 'u' or 'i' in args.mode:
                    scores_all = model.myparameters[0].mm(model.myparameters[1].t())
                    # scores_all[:,item_size] = -np.inf
                    scores = torch.gather(scores_all, 1, sampled_ids).to(args.device)

                if 'u' in args.mode:
                    ndcg = dcg_loss(scores[unique_u], scores_all[unique_u], labels[unique_u])
                    # ndcg = -torch.log(ndcg)
                    # print("ndcg:", -torch.log(ndcg))

                    mask_F = gender_label[unique_u]
                    mask_M = 1 - mask_F

                    mask_F = torch.from_numpy(mask_F).type(torch.FloatTensor).to(args.device)
                    mask_M = torch.from_numpy(mask_M).type(torch.FloatTensor).to(args.device)
                    pos_F = torch.tensor(np.where(mask_F.cpu() == 1)[0]).to(args.device)
                    pos_M = torch.tensor(np.where(mask_M.cpu() == 1)[0]).to(args.device)

                    ndcg_F = ndcg[pos_F]
                    ndcg_M = ndcg[pos_M]

                    # sum the matrix in column
                    ndcg_F = ndcg_F.sum(dim=0) / mask_F.sum()
                    ndcg_M = ndcg_M.sum(dim=0) / mask_M.sum()

                    loss['2'] = torch.abs(torch.log(1 + torch.abs(ndcg_F - ndcg_M))).sum()
                    # loss['2'] = torch.abs(torch.log(ndcg_F+0.5) - torch.log(ndcg_M+0.5)).sum()
                    # loss['2'] = loss['2'] * 20
                else:
                    loss['2'] = torch.tensor(0)

                if 'i' in args.mode:
                    ranks = ranker(scores[unique_u], scores_all[unique_u])

                    # print("ranks:", ranks.shape, ranks)
                    exposure = torch.pow(args.gamma, ranks)

                    prob = F.gumbel_softmax(scores[unique_u], tau=1, hard=False)
                    sys_exposure = exposure * prob

                    genre_top_mask = genre_mask[:, sampled_ids[unique_u].long()]
                    # print("genre_mask:", genre_mask.shape)
                    # print("genre_top_mask:", genre_top_mask.shape)

                    genre_exposure = torch.matmul(genre_top_mask.reshape(genre_num, -1), sys_exposure.reshape(-1, 1))
                    # print("genre_exposure:", genre_exposure.shape)
                    genre_exposure = genre_exposure / genre_exposure.sum()

                    loss['3'] = torch.abs(torch.log(1 + torch.abs(genre_exposure - target_exposure))).sum()
                    # loss['3'] = loss['3'] * args.batch_size
                    # loss['3'] = torch.abs(torch.log(genre_exposure+0.5) - torch.log(target_exposure+0.5)).sum()
                    # loss['3'] = loss['3'] * 20
                else:
                    loss['3'] = torch.tensor(0)

                # Use MOOP or not
                if args.moop:
                    # Copy the loss data. Average loss1 for calculating scale
                    for k in loss:
                        if k == '1':
                            loss_clone[k] = loss[k].clone() #/ args.batch_size
                        elif k == '2':
                            loss_clone[k] = loss[k].clone()
                        else:
                            loss_clone[k] = loss[k].clone()

                    for t in tasks:
                        # print(t)
                        optimizer.zero_grad()
                        loss_clone[t].backward(retain_graph=True)
                        loss_data[t] = loss_clone[t].item()

                        grads[t] = []
                        for param in model.myparameters:
                            if param.grad is not None:
                                tmp = Variable(param.grad.data.clone(), requires_grad=False).to(args.device)
                                tmp = tmp.flatten()
                                grads[t].append(tmp)

                    gn = gradient_normalizers(grads, loss_data, args.type)

                    for t in tasks:
                        if gn[t] == 0.0:
                            gn[t] += 1
                        for gr_i in range(len(grads[t])):
                            grads[t][gr_i] = grads[t][gr_i] / gn[t]

                    sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
                    for i, t in enumerate(tasks):
                        scale[t] = float(sol[i])
                else:
                    scale = {'1': args.scale1, '2': 1.0 - args.scale1, '3': 1.0 - args.scale1}

                batch_loss = 0
                for t in tasks:
                    batch_loss += loss[t] * scale[t]

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            print('\n'.join('\'{:s}\': {:.10f}'.format(k, scale[k]) for k in tasks))
            print('bpr_loss:{:.6f}, user_disparity:{:.6f}, item_disparity:{:.6f}'.format(loss['1'].item(),
                                                                                         loss['2'].item(),
                                                                                         loss['3'].item()))

            # end = time.time()
            # print("time:{:.2f}".format(end - start))

            ## save loss
            # loss_record[0].append(loss['1'].item())
            # loss_record[1].append(loss['2'].item())
            # loss_record[2].append(loss['3'].item())

            if (iter + 1) % args.every == 0:
                # with open(saved_record_path, "w") as fp:
                #     json.dump(loss_record, fp)
                model.eval()
                pred_list, score_matrix, raw_score_matrix = generate_pred_list(model, train_val_matrix, topk=200)
                precision, recall, MAP, ndcg = compute_metrics(test_user_list, pred_list, topk=20)
                logger.info(', '.join(str(e) for e in precision))
                logger.info(', '.join(str(e) for e in recall))
                logger.info(', '.join(str(e) for e in MAP))
                logger.info(', '.join(str(e) for e in ndcg))

                user_neg_items = neg_item_pre_sampling(train_val_matrix, num_neg_candidates=500)
                pre_samples = {'user_neg_items': user_neg_items}
                sampler = NegSampler(train_val_matrix, pre_samples, batch_size=args.batch_size, num_neg=1, n_workers=4)
        sampler.close()
    except KeyboardInterrupt:
        sampler.close()
        sys.exit()


    """
    Build top-200 recommendation list for later use
    """

    top200_id = pred_list
    top200_score = score_matrix
    merged_id = list(itertools.chain(*top200_id))
    pop_flat = list((pd.Series(merged_id)).map(popular_dict))
    top200_genre = np.array([pop_flat[x:x + 200] for x in range(0, len(pop_flat), 200)])

    pop_occurrence, ind_occurrence = statistics_occurrence(top200_id, popular_dict)

    """
    Fairness measurement
    """
    print("Fairness on user:")
    for k in [10, 20, 50, 100]:
        Fairness_user(test_user_list, pred_list, index_F, index_M, user_size, item_size, topk=k)
    print("Fairness on item:")
    for k in [10, 20, 50, 100]:
        Fairness_item(model, genre_mask, target_exposure, pred_list, user_size, topk=k, args=args)
    print("Gini index:")
    Gini(ind_occurrence)
    print("Popularity_rate:")
    Popularity_rate(pop_occurrence)
    print("Simpson_Diversity:")
    Simpson_Diversity(pop_occurrence)


    # print("top200_score:", top200_score.shape)
    # print("top200_genre:", top200_genre.shape)
    # print("top200_id:", top200_id.shape)



    logger.info('\n Parameters:')
    for arg, value in sorted(vars(args).items()):
        logger.info("%s: %r", arg, value)
    logger.info('\n')


if __name__ == '__main__':
    args = parse_args()
    print(args)
    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    # args.device = torch.device('cpu')
    print("device:", args.device)

    print("Data name:", args.data)
    args.batch_size = 1024

    # Preprocessing
    dataset, index_F, index_M, genre_mask, popular_dict = preprocessing(args)
    genre_mask = genre_mask.to(args.device)
    popular_tuple = OrderedDict(sorted(popular_dict.items()))
    popular_list = [x[1] for x in popular_tuple.items()]
    print("popular_tuple:", popular_tuple)
    print("popular_list:", popular_list)

    print("Number of females:", len(index_F))
    print("Number of males:", len(index_M))
    print("genre_mask:", genre_mask.shape)

    genre_num = genre_mask.shape[0]

    user_size, item_size = dataset['user_size'], dataset['item_size']
    train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['val_user_list'], dataset[
        'test_user_list']
    train_val_user_list = [i + j for i, j in zip(train_user_list, val_user_list)]

    all_list = [i + j + k for i, j, k in zip(train_user_list, val_user_list, test_user_list)]
    train_val_list = [i + j for i, j in zip(train_user_list, val_user_list)]

    # Build the observed raing matrix
    train_matrix, val_matrix, test_matrix = dataset['train_matrix'], dataset['val_matrix'], dataset['test_matrix']
    train_val_matrix = generate_rating_matrix(train_val_user_list, user_size, item_size)

    """only consider training and testing"""
    # Other statistics
    max_all_length = 0
    for i in range(len(all_list)):
        if len(all_list[i]) > max_all_length:
            max_all_length = len(all_list[i])
    print("max_all_length:", max_all_length)

    # for training.
    max_length = 0

    for i in range(len(train_val_user_list)):
        if len(train_val_user_list[i]) > max_length:
            max_length = len(train_val_user_list[i])
    print("max_train_val_length:", max_length)
    if args.data == 'ml-1m' or 'ml-100k':
        max_pos = max_length if max_length < 200 else 200
    elif args.data == 'lastfm':
        max_pos = max_length if max_length < 100 else 100
    print("max_pos:", max_pos)


    print()
    train(args)
