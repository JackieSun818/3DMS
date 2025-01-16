import random
import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch.nn as nn
from torch.utils.data import dataloader, TensorDataset, DataLoader
from utils_for_matching import *
#################基础设置################
LR = 0.0001
SEED = 3142
# VAL_SIZE = 7486
BATCH_SIZE = 128
EPOCH = 10
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


class GraphConvolution(nn.Module):
    def __init__(self, in_size, out_size,):
        super(GraphConvolution, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.FloatTensor(in_size, out_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, a):
        support = torch.mm(x, self.weight)  # X*W
        r = torch.mm(a, support)    # A*X*W
        return r


class DrugSMILESCoding(nn.Module):
    def __init__(self, hid_dim=Drug_SMILES_Input_Size, out_dim=Feature_Size, vocab_size=Atom_Dic_Length,
                 channel=DSC_Kernel_Num, kernel_size=DSC_Kernel_Size):
        super(DrugSMILESCoding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=hid_dim)
        self.conv1 = nn.Conv1d(hid_dim, channel, kernel_size, padding=kernel_size-1)
        self.conv2 = nn.Conv1d(channel, channel*2, kernel_size, padding=kernel_size-1)
        self.conv3 = nn.Conv1d(channel*2, channel*4, kernel_size, padding=kernel_size-1)
        self.act = nn.LeakyReLU(0.2)
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(channel*4, out_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.globalmaxpool(x)
        x = x.squeeze(-1)
        x = self.fc1(x)
        return x


class DrugPointCoding(nn.Module):
    def __init__(self, point_hid_dim=Drug_Point_Hidden_Size, point_output_dim=Feature_Size,
                 channel=DPC_Kernel_Num, kernel_size=DPC_Kernel_Size):
        super(DrugPointCoding, self).__init__()
        self.gcn1 = nn.Sequential(
            nn.Embedding(Atom_Point_Dict_Length, point_hid_dim),
            nn.Linear(point_hid_dim, point_hid_dim))
        self.gcn2 = nn.Linear(point_hid_dim, point_output_dim)

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

        self.conv = nn.Sequential(
            nn.Conv1d(point_output_dim, channel, kernel_size, padding=kernel_size - 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channel, channel * 2, kernel_size, padding=kernel_size - 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channel * 2, channel * 4, kernel_size, padding=kernel_size - 1),
            nn.LeakyReLU(0.2),
        )
        self.act = nn.LeakyReLU(0.2)
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(channel * 4, point_output_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, xd):
        d_d = xd[:, :, :xd.shape[1]]  # 距离
        d_c = xd[:, :, xd.shape[1]: xd.shape[1]*2]  # 距离
        d_a = xd[:, :, -1]  # 特征
        d_a = torch.squeeze(d_a).long()

        h1 = self.gcn1(d_a)
        d_shape = d_d.size()
        h_shape = h1.size()
        h1 = h1.view(d_shape[0], h_shape[-2], h_shape[-1])
        h1_1 = torch.einsum('ijk, ikp->ijp', d_d, h1)
        h1_2 = torch.einsum('ijk, ikp->ijp', d_c, h1)

        b, r, c = h1_1.shape
        h1_1 = self.act(h1_1).view(b, 1, r, c)
        h1_2 = self.act(h1_2).view(b, 1, r, c)
        h1_12 = torch.cat((h1_1, h1_2), dim=1)
        h1_c = self.conv1(h1_12).view(b, r, c)
        h1 = self.act(h1_c)

        h2 = self.gcn2(h1)
        d_shape = d_d.size()
        h_shape = h2.size()
        h2 = h2.view(d_shape[0], h_shape[-2], h_shape[-1])

        h2_1 = torch.einsum('ijk, ikp->ijp', d_d, h2)
        h2_2 = torch.einsum('ijk, ikp->ijp', d_c, h2)
        b, r, c = h2_1.shape
        h2_1 = self.act(h2_1).view(b, 1, r, c)
        h2_2 = self.act(h2_2).view(b, 1, r, c)
        h2_12 = torch.cat((h2_1, h2_2), dim=1)
        h2_c = self.conv2(h2_12).view(b, r, c)
        h2 = self.act(h2_c)
        h2 = self.act(h2)
        x = h2.permute(0, 2, 1)
        x = self.conv(x)
        x = self.globalmaxpool(x)
        x = x.squeeze(-1)
        x = self.fc1(x)
        return x


class DrugCoding(nn.Module):
    def __init__(self):
        super(DrugCoding, self).__init__()
        self.coding1 = DrugSMILESCoding()
        self.coding2 = DrugPointCoding()

    def forward(self, x_smiles, x_image):
        e_graph = self.coding1(x_smiles)
        e_image = self.coding2(x_image)

        return e_graph, e_image


class ProteinBertCoding(nn.Module):
    def __init__(self, bert_hid_dim=Protein_Bert_Hidden_Size, bert_output_dim=Feature_Size):
        super(ProteinBertCoding, self).__init__()
        self.seq_coding = nn.Sequential(
            nn.Linear(1024, bert_hid_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(bert_hid_dim, bert_output_dim),
            nn.Sigmoid(),
        )

    def forward(self, xp):
        ep = self.seq_coding(xp)
        return ep


class ProteinPointCoding(nn.Module):
    def __init__(self, point_hid_dim=Protein_Point_Hidden_Size, point_output_dim=Feature_Size,
                 channel=PPC_Kernel_Num, kernel_size=PPC_Kernel_Size):
        super(ProteinPointCoding, self).__init__()
        self.gcn1 = nn.Sequential(
            nn.Embedding(len(AA_Dict) + 1, point_hid_dim),
            nn.Linear(point_hid_dim, point_hid_dim))
        self.gcn2 = nn.Linear(point_hid_dim, point_output_dim)

        self.conv1 = nn.Sequential(
            nn.Conv1d(point_output_dim, channel, kernel_size, padding=kernel_size-1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channel, channel * 2, kernel_size, padding=kernel_size-1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channel * 2, channel * 4, kernel_size, padding=kernel_size-1),
            nn.LeakyReLU(0.2),
        )
        self.act = nn.LeakyReLU(0.2)
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(channel * 4, point_output_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, xp):
        p_t = xp[:, :, :xp.shape[1]]  # 拓扑
        p_a = xp[:, :, xp.shape[1]:]  # 特征
        p_a = torch.squeeze(p_a).long()
        h1 = self.gcn1(p_a)

        p_shape = p_t.size()
        h_shape = h1.size()
        h1 = h1.view(p_shape[0], h_shape[-2], h_shape[-1])

        h1 = torch.einsum('ijk, ikp->ijp', p_t, h1)
        h1 = self.act(h1)
        h2 = self.gcn2(h1)

        p_shape = p_t.size()
        h_shape = h2.size()
        h2 = h2.view(p_shape[0], h_shape[-2], h_shape[-1])

        h2 = torch.einsum('ijk, ikp->ijp', p_t, h2)
        h2 = self.act(h2)
        x = h2.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.globalmaxpool(x)
        x = x.squeeze(-1)
        x = self.fc1(x)
        return x


class ProteinCoding(nn.Module):
    def __init__(self):
        super(ProteinCoding, self).__init__()
        self.coding1 = ProteinBertCoding()
        self.coding2 = ProteinPointCoding()

    def forward(self, x_bert, x_point):
        e_bert = self.coding1(x_bert)
        e_point = self.coding2(x_point)

        return e_bert, e_point


class PreNetMLP(nn.Module):
    def __init__(self, smiles_output_dim=Feature_Size, bert_output_dim=Feature_Size):
        super(PreNetMLP, self).__init__()
        self.d_c = DrugCoding()
        self.p_c = ProteinCoding()
        self.fc1 = nn.Linear(smiles_output_dim + bert_output_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)
        self.act1 = nn.LeakyReLU(0.2)
        self.act2 = nn.Tanh()

    def forward(self, d_s, d_i, p_b, p_p):
        eds, edi = self.d_c(d_s, d_i)
        epb, epp = self.p_c(p_b, p_p)
        e = torch.cat((eds, epb), dim=1)
        s0 = self.fc1(e)
        a0 = self.act1(s0)
        s1 = self.fc2(a0)
        a1 = self.act2(s1)
        s2 = self.fc3(a1)
        return s2


def seed_torch(seed):
    random.seed()
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    task = "drugbank"
    fold = 0
    with open('dataset/' + task + '/result/CV5/test_cv' + str(fold) + '.csv') as f3:
        test_data = f3.readlines()
    num_sample = len(test_data)
    drug_idx_train, protein_idx_train, label_train = data_preparation(test_data, task)
    # 将所有的药物和蛋白所需数据读入内存中方便后续快速取用，确保针对每种药物/蛋白只处理一次
    drug_smiles_data, drug_points_data = data_preparation_drug_all(task)
    protein_bert_data, protein_point_data = data_preparation_protein_all(task)

    dataset = TensorDataset(drug_idx_train, protein_idx_train, label_train)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    model = PreNetMLP()
    model.load_state_dict(torch.load('models/Ablation_only_linear_drugbank.pth'))
    # print(model)
    if torch.cuda.is_available():
        model = model.cuda()
    pre0 = []
    pre1 = []
    lab = []
    embedding = []
    for step, data in enumerate(dataloader):
        batch_d_idx, batch_p_idx, batch_y = data
        batch_xd1, batch_xd2 = data_preparation_drug(drug_smiles_data, drug_points_data, batch_d_idx)
        batch_xp1, batch_xp2 = data_preparation_protein(protein_bert_data, protein_point_data, batch_p_idx)

        if torch.cuda.is_available():
            batch_xd1 = batch_xd1.cuda()
            batch_xd2 = batch_xd2.cuda()
            batch_xp1 = batch_xp1.cuda()
            batch_xp2 = batch_xp2.cuda()
            batch_y = batch_y.cuda()
        # print(epoch, step)
        batch_pre = model(batch_xd1, batch_xd2, batch_xp1, batch_xp2)
        # embedding += hid.detach().cpu().numpy().tolist()
        pre0 += batch_pre.detach().cpu().numpy()[:, 0].tolist()
        pre1 += batch_pre.detach().cpu().numpy()[:, 1].tolist()
        lab += batch_y.detach().cpu().numpy().tolist()
        print(str(step) + ' / ' + str(int(num_sample/BATCH_SIZE+1)) + '    is already!' )

    result = np.vstack((lab, pre0, pre1))
    np.savetxt('models/结果/Ablation_only_linear_' + task + '_cv' + str(fold) + '.csv', result, delimiter=',', fmt='%2f')




