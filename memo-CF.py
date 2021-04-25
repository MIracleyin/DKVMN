#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/4/24 上午9:14
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : memo-CF.py
import torch
import random
import argparse
from model import MODEL
from utils import *
import numpy as np
from data_loader import DATA

display_tensorshape(False)
seed = 73
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# 使用testset放入model中，生成memo_value
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the gpu will be used, e.g "0,1,2,3"')
    parser.add_argument('--max_iter', type=int, default=30, help='number of iterations')
    parser.add_argument('--decay_epoch', type=int, default=20, help='number of iterations')
    parser.add_argument('--test', type=bool, default=False, help='enable testing')
    parser.add_argument('--train_test', type=bool, default=True, help='enable testing')
    parser.add_argument('--show', type=bool, default=True, help='print progress')
    parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
    parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.75, help='learning rate decay')
    parser.add_argument('--final_lr', type=float, default=1E-5,
                        help='learning rate will not decrease after hitting this threshold')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
    parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
    parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')

    dataset = 'assist2009_updated'

    if dataset == 'assist2009_updated':
        parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
        parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
        parser.add_argument('--qa_embed_dim', type=int, default=200, help='answer and question embedding dimensions')
        parser.add_argument('--memory_size', type=int, default=20, help='memory size')
        parser.add_argument('--n_question', type=int, default=110, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='./data/assist2009_updated', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2009_updated', help='data set name')
        parser.add_argument('--load', type=str, default='data/assist2009_updated', help='model file to load')
        parser.add_argument('--save', type=str, default='data/assist2009_updated/model', help='path to save model')

    elif dataset == 'STATICS':
        parser.add_argument('--batch_size', type=int, default=10, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
        parser.add_argument('--qa_embed_dim', type=int, default=100, help='answer and question embedding dimensions')
        parser.add_argument('--memory_size', type=int, default=50, help='memory size')
        parser.add_argument('--n_question', type=int, default=1223,
                            help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=6, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='./data/STATICS', help='data directory')
        parser.add_argument('--data_name', type=str, default='STATICS', help='data set name')
        parser.add_argument('--load', type=str, default='STATICS', help='model file to load')
        parser.add_argument('--save', type=str, default='STATICS', help='path to save model')

    params = parser.parse_args()
    model = torch.load(params.save + "/best.pt")
    print(params)
    ## 读取test数据 作为推荐数据集合
    dat = DATA(n_question=params.n_question, seqlen=params.seqlen, separate_char=',')
    test_data_path = params.data_dir + "/" + params.data_name + "_test.csv"
    test_q_data, test_qa_data, test_id = dat.load_data(test_data_path)
    # 制作 user-q-qa字典 用于推荐以及SR计算
    id_q_qa = id_q_qa_dict(test_id, test_q_data, test_qa_data)

    ## 读取每个用户的知识矩阵，协同过滤后得到相似用户
    km = knowledge_matrix(model, params, test_id, test_q_data, test_qa_data)
    user_distance = user_distance_matrix(km, params)
    user_top3 = user_topk(user_distance, 3)

    ## 使用相似用户，对于该用户推荐，得到推荐习题
    rec_id = test_id[100]
    recom_q_id = user_recom_q(rec_id, id_q_qa, user_top3)
    print(recom_q_id)


if __name__ == '__main__':
    main()
