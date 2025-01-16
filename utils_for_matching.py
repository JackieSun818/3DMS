import os
import numpy as np
import math
from tqdm import *
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, precision_score, recall_score, auc
# import cv2
import torch
import sys


#################基础设置################
LR = 0.0001
VAL_SIZE = 7486
BATCH_SIZE = 64
EPOCH = 20
Feature_Size = 256   # [256, 512]
Alpha_d = 1
Alpha_p = 1

###############药物编码模块###############
# SMILES_Coding
DSC_Kernel_Num = 32
DSC_Kernel_Size = 8
Drug_SMILES_Input_Size = 128      # [128, 256]

# Image_Coding
Drug_Point_Hidden_Size = 512   # [128, 256, 512]
DPC_Kernel_Num = 32
DPC_Kernel_Size = 8   # [8, 16]

###############蛋白编码模块###############
# Bert_Coding
Protein_Bert_Hidden_Size = 512

#Point_Coding
Protein_Point_Hidden_Size = 512   # [128, 256, 512]
PPC_Kernel_Num = 32
PPC_Kernel_Size = 8   # [8, 16]

###############数据处理设置################
Drug_Max_Lengtgh = 100
Protein_Max_Lengtgh = 1024
AA_Dict = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B', 'Z']
Protein_Dic_Length = 23
Atom_Point_Dict_Length = 79
atom_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, "1": 35, "0": 3,
            "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, "9": 39, "8": 7, "=": 40, "A": 41,
            "@": 8, "C": 42, "B": 9, "E": 43, "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46,
            "M": 47, "L": 13, "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
            "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, "b": 21, "e": 57,
            "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, "l": 25, "o": 61, "n": 26, "s": 62,
            "r": 27, "u": 63, "t": 28, "y": 64}

Atom_Dic_Length = 64


def laplacians(A):
    n = A.shape[0]                # m = n = 973
    m = A.shape[1]
    D = np.sum(A, axis=1)         # D是973维的向量
    A_L = np.zeros(A.shape)
    for i in range(n):
        for j in range(m):
            if i == j and D[i] != 0:
                A_L[i, j] = 1
            elif i != j and A[i, j] != 0:
                A_L[i, j] = (-1)/math.sqrt(D[i] * D[j])
            else:
                A_L[i, j] = 0
    return A_L


def max_min_normalize(a):                              #矩阵归一化
    sum_of_line = np.sum(a, axis=1)
    line = a.shape[0]
    row = a.shape[1]
    i = 0
    a = a.tolist()
    a_n = []
    for i in range(len(a)):
        if sum_of_line[i] != 0:
            max = np.max(np.array(a[i]))
            min = np.min(np.array(a[i]))
            t = []
            for j in range(len(a[0])):
                t.append((a[i][j] - min) / (max - min))
        else:
            t = []
            for j in range(len(a[0])):
                t.append(0)
        a_n.append(t)
    return np.array(a_n)


def line_normalize(A):                        #行归一化
    sum_of_line = np.sum(A, axis=1)
    line = A.shape[0]
    row = A.shape[1]
    i = 0
    while i < line:
        j = 0
        while j < row:
            if sum_of_line[i] != 0:
                A[i, j] = A[i, j] / sum_of_line[i]
            j = j + 1
        i = i + 1
    return A


def data_preparation(data, task):
    print('drug/protein index preparation....')
    with open('dataset/' + task + '/drug_SMILES.csv') as f1:
        drug_info = f1.readlines()
    drug_list = []
    for i in drug_info:
        drug_list.append(i.split(',')[0])  # drug_info 药物名称列表，按照在药物smiles文件中出现的顺序

    with open('dataset/' + task + '/normalized_seq_bert.csv') as f2:
        protein_info = f2.readlines()
    protein_list = []
    for i in protein_info:
        protein_list.append(i.split(',')[0])

    protein_idx = []
    drug_idx = []
    label = []
    for i in tqdm(data):
        t = i.split(',')
        if os.path.exists('dataset/' + task + '/Drug_Point_Graph/' + str(t[0]) + '.txt') and os.path.exists('dataset/' + task + '/protein_image_3D_normal/' + str(t[1]) + '.txt'):
            drug_idx.append(drug_list.index(t[0]))  # 找到训练集中药物对应的id，以此构建药物id列表
            protein_idx.append(protein_list.index(t[1]))
            label.append(t[2].replace('\n', ''))
    drug_idx_tensor = torch.from_numpy(np.array(drug_idx, dtype=int))
    protein_idx_tensor = torch.from_numpy(np.array(protein_idx, dtype=int))
    label_tensor = torch.from_numpy(np.array(label, dtype=int)).long()

    return drug_idx_tensor, protein_idx_tensor, label_tensor


def data_preparation_drug_all(task):
    # 这里要计算全部药物的idx索引，其实就是每个药物在drug_SMILES.txt文件中出现的顺序
    print('drug data preparation....')
    drug_smiles_path = '/drug_SMILES.csv'
    with open('dataset/' + task + drug_smiles_path) as f1:
        drug_info = f1.readlines()
    drug_list = []
    drug_idx = 0
    for i in drug_info:
        drug_list.append(drug_idx)  # drug_list：药物列表，顺序就是按照在药物smiles文件中出现的顺序
        drug_idx += 1

    idx = torch.from_numpy(np.array(drug_list, dtype=int))

    with open('dataset/' + task + drug_smiles_path) as f1:
        drug_info = f1.readlines()
    drug_list = []
    for i in drug_info:
        drug_list.append([i.split(',')[0], i.split(',')[1].replace('\n', '')])
    points = []
    smiles = []
    for i in trange(idx.shape[0]):
        d_name = drug_list[int(idx[i])][0]
        d_smiles = drug_list[int(idx[i])][1]
        s = np.zeros(Drug_Max_Lengtgh)
        for j in range(min(len(d_smiles), Drug_Max_Lengtgh)):
            if d_smiles[j] in atom_dict:
                s[j] = atom_dict[d_smiles[j]]
        smiles.append(s.tolist())
        if d_name + '.txt' not in os.listdir('dataset/' + task + '/Drug_Point_Graph/'):
            d_point_ori = np.array([1,1, 1])
        else:
            d_point_ori = np.loadtxt('dataset/' + task + '/Drug_Point_Graph/' + d_name + '.txt')
        try:
            aaa = d_point_ori.shape[1]
        except:
            d_point_ori = d_point_ori.reshape(1, 3)
        point_size = d_point_ori.shape[0]
        if d_point_ori.shape[0] >= Drug_Max_Lengtgh:

            d1 = d_point_ori[0:Drug_Max_Lengtgh, 0:Drug_Max_Lengtgh]
            d2 = d_point_ori[0:Drug_Max_Lengtgh, point_size: point_size+Drug_Max_Lengtgh]
            d3 = d_point_ori[0:Drug_Max_Lengtgh, -1]
            d_point = np.hstack((d1, d2))
            d_point = np.hstack((d_point, d3.reshape(-1, 1)))
        else:
            d_point = np.zeros((Drug_Max_Lengtgh, Drug_Max_Lengtgh * 2 + 1))
            try:
                d_point[:point_size, :point_size] = d_point_ori[:, :point_size]
            except:
                print('1')
            d_point[:point_size, Drug_Max_Lengtgh: Drug_Max_Lengtgh + point_size] = d_point_ori[:, point_size: point_size*2]
            d_point[:point_size, -1] = d_point_ori[:, -1]
        points.append(d_point.tolist())

    smiles = torch.from_numpy(np.array(smiles, dtype=int)).long()
    points = torch.from_numpy(np.array(points, dtype=float)).float()
    return smiles, points


def data_preparation_drug(drug_smiles_data, drug_points_data, idx):  # torch、torch、torch
    idx = idx.long()
    smiles = drug_smiles_data[idx, :]
    points = drug_points_data[idx, :]
    return smiles, points


def data_preparation_protein_all(task):
    # 这里要计算全部蛋白的idx索引，其实就是每个蛋白在normalized_seq_bert.csv文件中出现的顺序
    print('protein data preparation....')
    with open('dataset/' + task + '/normalized_seq_bert.csv') as f2:
        protein_info = f2.readlines()
    protein_list = []
    protein_idx = 0
    for i in protein_info:
        protein_list.append(protein_idx)
        protein_idx += 1

    idx = torch.from_numpy(np.array(protein_list, dtype=int))

    if task == "bindingdb" or task == "dtinet":
        protein_seq = np.loadtxt('dataset/' + task + '/protein_seq.csv', delimiter=',', dtype=str)
    else:
        protein_seq = np.loadtxt('dataset/' + task + '/protein_seq.txt', delimiter=',', dtype=str)
    protein_info = np.loadtxt('dataset/' + task + '/normalized_seq_bert.csv', delimiter=',', dtype=str)
    batch_bert = []
    batch_point = []
    for i in trange(idx.shape[0]):
        batch_bert.append(protein_info[int(idx[i]), 1:].tolist())  # 因为参数中提供的id就是按照文件中存放的顺序来提供的，所以这里可以直接用它来进行索引
        pro_name = protein_info[int(idx[i]), 0]
        if pro_name + '.txt' not in os.listdir('dataset/' + task + '/protein_image_3D_normal/'):  # 如果没有对应的点云文件的话，直接赋值一个零均，反正后边不用，无所谓
            pro_topology_ori = np.array([[0, 0], [0, 0]])
        else:
            pro_topology_ori = np.loadtxt('dataset/' + task + '/protein_image_3D_normal/' + pro_name + '.txt')

        if pro_topology_ori.shape[0] >= Protein_Max_Lengtgh:  # 设置图的最大取值为1024，多的部分直接忽略，少的部分补零
            pro_topology = pro_topology_ori[:Protein_Max_Lengtgh, : Protein_Max_Lengtgh]
        else:
            pro_topology = np.zeros((Protein_Max_Lengtgh, Protein_Max_Lengtgh))
            pro_topology[0: pro_topology_ori.shape[0], 0: pro_topology_ori.shape[1]] = pro_topology_ori
        pro_feature = np.zeros((Protein_Max_Lengtgh, 1))  # 残基类型部分进行读取，紧接着转换为字典对应的下标值，字典中没有的残基类型默认规定为字典的长度值
        for aa in range(min(len(protein_seq[int(idx[i]), 1]), Protein_Max_Lengtgh)):
            if protein_seq[int(idx[i]), 1][aa] in AA_Dict:
                pro_feature[aa] = AA_Dict.index(protein_seq[int(idx[i]), 1][aa])
            else:
                pro_feature[aa] = len(AA_Dict)
        pro_graph = np.hstack((pro_topology, pro_feature))  # 将距离图和类型列表压在一起
        batch_point.append(pro_graph)
    batch_bert = torch.from_numpy(np.array(batch_bert, dtype=float)).float()
    batch_point = torch.from_numpy(np.array(batch_point, dtype=float)).float()
    return batch_bert, batch_point


def data_preparation_protein(protein_bert_data, protein_point_data, idx):  # 将需要的蛋白质id列表进行传入
    idx = idx.long()
    batch_bert = protein_bert_data[idx, :]
    batch_point = protein_point_data[idx, :]
    return batch_bert, batch_point


def val_evalute(y_pred, y_label):
    y_pred = np.array(y_pred, dtype=float)
    y_label = np.array(y_label, dtype=int)
    fpr, tpr, thresholds = roc_curve(y_label, y_pred)
    precision = tpr / (tpr + fpr + 0.00001)
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    thred_optim = thresholds[5:][np.argmax(f1[5:])]
    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
    auc_k = auc(fpr, tpr)
    aupr = average_precision_score(y_label, y_pred)
    cm1 = confusion_matrix(y_label, y_pred_s)
    total1 = sum(sum(cm1))
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    sensitivity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    specificity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    return auc_k, aupr, accuracy1, sensitivity1, specificity1


