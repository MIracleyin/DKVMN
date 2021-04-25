import json
import os
import torch.nn.init
import numpy as np
import pandas as pd
import math
from collections import defaultdict


def varible(tensor, gpu):
    if gpu >= 0:
        return torch.autograd.Variable(tensor).cuda()
    else:
        return torch.autograd.Variable(tensor)


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def save_checkpoint(state, track_list, filename):
    with open(filename + '.json', 'w') as f:
        json.dump(track_list, f)
    torch.save(state, filename + '.model')


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def generate_dir(work_dir):
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)


def display_tensorshape(is_display=True):
    # 只显示 tensor shape
    if is_display:
        old_repr = torch.Tensor.__repr__

        def tensor_info(tensor):
            return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(
                tensor)

        torch.Tensor.__repr__ = tensor_info


def id_q_qa_dict(ids, qs, qas):
    id_q_qa = defaultdict(list)
    for id, q, qa in zip(ids, qs, qas):
        id_q_qa[id].append(q)
        id_q_qa[id].append(qa)
    return id_q_qa


def self_cosine_distance(a, b):
    return torch.cosine_similarity(a, b)


def self_euclid_distance(a, b):
    return torch.dist(a, b)


def knowledge_matrix(model, params, id, q_data, qa_data):
    parallel = len(id)
    # 一次性加载
    knowledge_dict = {}
    N = int(math.floor(len(id) / parallel))  # inference 一条一条加载 一次性全加载解决id和matrix匹配问题
    model.eval()

    for idx in range(N):
        q_one_seq = q_data[idx * parallel: (idx + 1) * parallel, :]
        qa_one_seq = qa_data[idx * parallel: (idx + 1) * parallel, :]
        target = qa_data[idx * parallel: (idx + 1) * parallel, :]

        target = (target - 1) / params.n_question
        target = np.floor(target)

        input_q = varible(torch.LongTensor(q_one_seq), params.gpu)  # shape 1,200
        input_qa = varible(torch.LongTensor(qa_one_seq), params.gpu)  # shape 1,200
        target = varible(torch.FloatTensor(target), params.gpu)  # shape 1,200

        target_to_1d = torch.chunk(target, parallel, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(parallel)], 1)
        target_1d = target_1d.permute(1, 0)
        _, _, _, memory = model.forward(input_q, input_qa, target_1d)
    memory_list = torch.chunk(memory, parallel, 0)
    # knowledge_dict = {single_id:  for single_id, memory_matrix in zip(id, memory_list)}
    for single_id, memory_matrix in zip(id, memory_list):
        if single_id in knowledge_dict.keys():
            knowledge_dict[single_id] += memory_matrix.squeeze()
        else:
            knowledge_dict.update({single_id: memory_matrix.squeeze()})
    return knowledge_dict


def user_distance_matrix(knowledge_matrix, params):
    """
    用于计算用户和用户之间的知识水平的欧式距离
    :param knowledge_matrix: 从模型提取的记忆力矩阵
    :param params:
    :return: 一个key为id，value为子字典，子字典key为id，value为距离
    """
    user_distance = {}
    for id_x, knowledge_x in knowledge_matrix.items():
        user_distance_y = {}
        for id_y, knowledge_y in knowledge_matrix.items():
            # 把自己和自己的距离记做正无穷
            distance = varible(torch.tensor(float('inf')), params.gpu) if id_x == id_y else torch.dist(knowledge_x,
                                                                                                       knowledge_y)
            user_distance_y[id_y] = distance.cpu().detach().numpy()  # 把距离放到内存上，去除梯度，转为np
            # user_distance[id_x].append({id_y:distance})
        user_distance[id_x] = user_distance_y
    return user_distance


def user_topk(user_dic, K):
    """
    使用用户之间知识水平的欧氏距离，生成用户最接近的K个用户
    :param user_dic: 知识水平距离字典
    :param K: 最相近的K个用户
    :return: 返回该用户最接近的K个用户id
    """
    user_recom_dic = {}
    for user_id_x, user_distance_row in user_dic.items():
        user_distance_row = pd.Series(user_distance_row).sort_values()
        user_recom_dic[user_id_x] = user_distance_row.keys()[:K]
    return user_recom_dic


def user_recom_q(user_id, id_q_qa_dic, user_recom_topn, rec_q_len=-1):
    """
    返回推荐的习题列表
    :param user_id: 需要推荐的用户id
    :param id_q_qa_dic: 根据数据集构造用户id-问题-回答词典
    :param user_recom_topn: 相似用户词典
    :return: 推荐的习题
    """
    like_users = user_recom_topn[user_id]
    self_q = id_q_qa_dic[user_id][0]
    recomd_q = []
    for like_user in like_users:
        temp_qs = id_q_qa_dic[like_user][0]
        for temp_q in temp_qs:
            if temp_q not in self_q:
                recomd_q.append(temp_q)
    recomd_q = list(set(recomd_q))
    if rec_q_len == -1:
        return recomd_q
    elif len(recomd_q) >= rec_q_len:
        return recomd_q[:rec_q_len]
    else:
        print(f"可推荐习题数量达不到{rec_q_len}")
        return recomd_q
    return list(set(recomd_q))


def partition_arg_topk(array, K, axis=0):
    a_part = np.argpartition(array, -K, axis=axis)[-K: len(array)]
    if axis == 0:
        row_index = np.arange(array.shape[1 - axis])
        a_sec_argsort_K = np.argsort(array[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index][::-1]
    else:
        column_index = np.arange(array.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(array[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K][::-1]
