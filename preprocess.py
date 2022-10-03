import os
import gzip
import json
import math
import random
import pickle
import pprint
import argparse
import torch

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from collections import Counter
from sklearn.model_selection import train_test_split

from copy import deepcopy

class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError


class MovieLens100k(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath_rate = os.path.join(data_dir, 'u.data')
        self.fpath_gender = os.path.join(data_dir, 'u.user')
        self.fpath_genre = os.path.join(data_dir, 'u.item')

    def load(self):
        # Load data
        df_rate = pd.read_csv(self.fpath_rate,
                              sep='\t',
                              engine='python',
                              names=['user', 'item', 'rate', 'time'],
                              usecols=['user', 'item', 'rate'])

        df_gender = pd.read_csv(self.fpath_gender,
                                sep='|',
                                engine='python',
                                names=['user', 'age', 'gender', 'occupation', 'zip'],
                                usecols=['user', 'gender'])

        df_rate['user'] = df_rate['user'].astype(str)
        df_gender['user'] = df_gender['user'].astype(str)

        df = pd.merge(df_rate, df_gender, on='user')
        df = df.dropna(axis=0, how='any')

        # TODO: Remove negative rating?
        df = df[df['rate'] > 3]
        df = df.reset_index().drop(['index'], axis=1)

        return df




class MovieLens1M(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath_rate = os.path.join(data_dir, 'ratings.dat')
        self.fpath_user = os.path.join(data_dir, 'users.dat')
        self.fpath_item = os.path.join(data_dir, 'movies.dat')

    def load(self):
        # Load data
        df_rate = pd.read_csv(self.fpath_rate,
                              sep='::',
                              engine='python',
                              names=['user', 'item', 'rate', 'time'],
                              usecols=['user', 'item', 'rate'])
        df_user = pd.read_csv(self.fpath_user,
                              sep='::',
                              engine='python',
                              names=['user', 'gender', 'age', 'occupation', 'Zip-code'],
                              usecols=['user', 'gender'])

        df = pd.merge(df_rate, df_user, on='user')
        df = df.dropna(axis=0, how='any')
        # TODO: Remove negative rating?
        df = df[df['rate'] > 3]
        df = df.reset_index().drop(['index'], axis=1)

        return df




class LastFM(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath_rate = os.path.join(data_dir, 'usersha1-artmbid-artname-plays.tsv')
        self.fpath_user = os.path.join(data_dir, 'usersha1-profile.tsv')

    def load(self):
        # Load data
        df_rate = pd.read_csv(self.fpath_rate,
                              sep='\t',
                              names=['user', 'item', 'artist', 'rate'],
                              usecols=['user', 'item', 'rate']
                              )
        df_user = pd.read_csv(self.fpath_user,
                                sep='\t',
                                names=['user', 'gender', 'age', 'country', 'signup'],
                                usecols=['user', 'gender'])

        df_rate['user'] = df_rate['user'].astype(str)
        df_user['user'] = df_user['user'].astype(str)

        df = pd.merge(df_rate, df_user, on='user')
        df = df.dropna(axis=0, how='any')

        return df



def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    # print("df:", df[column_name])
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict


def create_user_list(df, user_size):
    user_list = [list() for u in range(user_size)]
    for row in df.itertuples():
        user_list[row.user].append(row.item)

    return user_list


def split_data_randomly(user_records, val_ratio, test_ratio, seed=0):
    train_set = []
    test_set = []
    val_set = []
    for user_id, item_list in enumerate(user_records):
        tmp_train_sample, tmp_test_sample = train_test_split(item_list, test_size=test_ratio, random_state=seed)

        if val_ratio:
            tmp_train_sample, tmp_val_sample = train_test_split(tmp_train_sample, test_size=val_ratio,
                                                                random_state=seed)

        if val_ratio:
            train_sample = []
            for place in item_list:
                if place not in tmp_test_sample and place not in tmp_val_sample:
                    train_sample.append(place)

            val_sample = []
            for place in item_list:
                if place not in tmp_test_sample and place not in tmp_train_sample:
                    val_sample.append(place)

            test_sample = []
            for place in tmp_test_sample:
                if place not in tmp_train_sample and place not in tmp_val_sample:
                    test_sample.append(place)

            train_set.append(train_sample)
            val_set.append(val_sample)
            test_set.append(test_sample)

        else:
            train_sample = []
            for place in item_list:
                if place not in tmp_test_sample:
                    train_sample.append(place)

            test_sample = []
            for place in tmp_test_sample:
                if place not in tmp_train_sample:
                    test_sample.append(place)

            train_set.append(train_sample)
            test_set.append(test_sample)

    return train_set, test_set, val_set


def split_train_test(df, user_size, item_size, val_ratio=0.1, test_ratio=0.1):
    data_records = create_user_list(df, user_size)

    train_set, test_set, val_set = split_data_randomly(data_records, val_ratio=val_ratio, test_ratio=test_ratio)
    train_matrix = generate_rating_matrix(train_set, user_size, item_size)
    test_matrix = generate_rating_matrix(test_set, user_size, item_size)
    val_matrix = generate_rating_matrix(val_set, user_size, item_size)

    return train_matrix, test_matrix, val_matrix, train_set, test_set, val_set

# def split_data_sequentially(self, user_records, test_radio=0.2):
#         train_set = []
#         test_set = []
#
#         for item_list in user_records:
#             len_list = len(item_list)
#             num_test_samples = int(math.ceil(len_list * test_radio))
#             train_sample = []
#             test_sample = []
#             for i in range(len_list - num_test_samples, len_list):
#                 test_sample.append(item_list[i])
#
#             for place in item_list:
#                 if place not in set(test_sample):
#                     train_sample.append(place)
#
#             train_set.append(train_sample)
#             test_set.append(test_sample)
#
#         # train_val_set, test_set = self.split_data_sequentially(user_records, test_radio=0.2)
#         # train_set, val_set = self.split_data_sequentially(train_val_set, test_radio=0.1)
#
#         return train_set, test_set


def create_pair(user_list):
    pair = []
    for user, item_list in enumerate(user_list):
        pair.extend([(user, item) for item in item_list])
    return pair


def generate_rating_matrix(train_set, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, article_list in enumerate(train_set):
        for article in article_list:
            row.append(user_id)
            col.append(article)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def gender_index(df):
      gender_dic = df.groupby('user')['gender'].apply(list).to_dict()
      index_F = []
      index_M = []
      for i in range(0, len(gender_dic)):
        if 'f' in gender_dic[i] or 'F' in gender_dic[i]:
          index_F.append(i)
        else:
          index_M.append(i)
      index_F = np.array(index_F)
      index_M = np.array(index_M)
      return index_F, index_M


def popularity_index(df):
    count = Counter(df['item'])
    occur = count.most_common()
    length = len(occur)
    item_size = len(set(df['item']))

    popular = {}
    # for i in range(item_size):
    #     if count[i] > occur[int(0.2 * len(occur))][1]:
    #         popular[i] = 5
    #     elif count[i] > occur[int(0.4 * len(occur))][1]:
    #         popular[i] = 4
    #     elif count[i] > occur[int(0.6 * len(occur))][1]:
    #         popular[i] = 3
    #     elif count[i] > occur[int(0.8 * len(occur))][1]:
    #         popular[i] = 2
    #     else:
    #         popular[i] = 1
    #
    for i, v in enumerate(occur):
        if v[1] >= occur[int(0.2 * length)][1]:
            popular[v[0]] = 5
        elif v[1] >= occur[int(0.4 * length)][1]:
            popular[v[0]] = 4
        elif v[1] >= occur[int(0.6 * length)][1]:
            popular[v[0]] = 3
        elif v[1] >= occur[int(0.8 * length)][1]:
            popular[v[0]] = 2
        else:
            popular[v[0]] = 1


    genre_size = 5
    genre_mask = torch.zeros(genre_size, item_size)
    for i in range(item_size):
        for k in range(1, genre_size + 1):
            if popular[i] == k:
                genre_mask[k - 1][i] = 1

    print("genre_mask:", genre_mask, genre_mask.sum(dim=1))

    return genre_mask, popular


def remove_infrequent_items(data, min_counts=5):
    df = deepcopy(data)
    counts = df['item'].value_counts()
    df = df[df['item'].isin(counts[counts >= min_counts].index)]

    print("items with < {} interactoins are removed".format(min_counts))
    # print(df.describe())
    return df

def remove_infrequent_users(data, min_counts=10):
    df = deepcopy(data)
    counts = df['user'].value_counts()
    df = df[df['user'].isin(counts[counts >= min_counts].index)]

    print("users with < {} interactoins are removed".format(min_counts))
    # print(df.describe())
    return df




def preprocessing(args):

    data_dir = os.path.join('data', args.data)

    if args.data == 'ml-1m':
        df = MovieLens1M(data_dir).load()
        df = df.groupby('user').filter(lambda x: len(x) <= 500)
        for i in range(5):
            df = df.groupby('user').filter(lambda x: len(x) >= 10)
            df = df.groupby('item').filter(lambda x: len(x) >= 10)
            df = df.reset_index().drop(['index'], axis=1)
    elif args.data == 'ml-100k':
        df = MovieLens100k(data_dir).load()
    elif args.data == 'lastfm':
        df = LastFM(data_dir).load()
        df = remove_infrequent_users(df, 60)
        df = remove_infrequent_items(df, 20)
        unique_data = df.groupby('user')['item'].nunique()
        df = df.loc[df['user'].isin(unique_data[unique_data >= 20].index)]
    else:
        raise NotImplementedError


    df, user_mapping = convert_unique_idx(df, 'user')
    df, item_mapping = convert_unique_idx(df, 'item')
    df = df.reset_index().drop(['index'], axis=1)
    print("df:", df)
    print('Complete assigning unique index to user and item')

    user_size = len(df['user'].unique())
    item_size = len(df['item'].unique())

    print("user_size:", user_size)
    print("item_size:", item_size)

    train_matrix, test_matrix, val_matrix, train_user_list, test_user_list, val_user_list\
        = split_train_test(df, user_size, item_size, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    print('Complete spliting items for training, validation, and testing')

    train_pair = create_pair(train_user_list)
    print('Complete creating pair')


    dataset = {'user_size': len(user_mapping), 'item_size': len(item_mapping),
               'user_mapping': user_mapping, 'item_mapping': item_mapping,
               'train_matrix': train_matrix, 'val_matrix': val_matrix, 'test_matrix': test_matrix,
               'train_user_list': train_user_list, 'val_user_list': val_user_list, 'test_user_list': test_user_list,
               'train_pair': train_pair}

    index_F, index_M = gender_index(df)
    pop_mask, popular_dict = popularity_index(df)

    return dataset, index_F, index_M, pop_mask, popular_dict

