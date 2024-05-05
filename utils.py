import numpy as np
import torch
from sklearn.metrics import roc_curve, precision_recall_curve, auc, matthews_corrcoef, accuracy_score, f1_score

def scaley(ymat):
    return (ymat-ymat.min())/ymat.max()

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def load_data(data):
    path = 'Dataset'+str(data)
    dis = np.loadtxt(path + '/dis_ss.csv', delimiter=",")
    dru = np.loadtxt(path + '/drug_ss.csv', delimiter=",")
    ddi = np.loadtxt(path + '/didr.csv', delimiter=",")
    ddit = torch.from_numpy(ddi).float()
    dis_sim = torch.from_numpy(dis)             # 把数组转换成张量，且二者共享内存，对张量进行修改
    drug_sim = torch.from_numpy(dru)            # 比如重新赋值，那么原始数组也会相应发生改变。
    dis_sim = dis_sim.float()
    drug_sim = drug_sim.float()
    return dis_sim, drug_sim, ddit

def neighborhood(feat, k):
    # compute C
    featprod = np.dot(feat.T, feat)
    smat = np.tile(np.diag(featprod), (feat.shape[1], 1))
    dmat = smat + smat.T - 2*featprod
    dsort = np.argsort(dmat)[:, 1:k]
    C = np.zeros((feat.shape[1], feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i, j] = 1.0
    return C

def normalized(wmat):
    deg = np.diag(np.sum(wmat, axis=0))
    degpow = np.power(deg, -0.5)
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow, wmat), degpow)
    return W

def norm_adj(A):
    C = neighborhood(A.T, k=10)
    #A = np.asarray(A)
    norm_adj = normalized(C.T*C+np.eye(C.shape[0]))
    g = torch.from_numpy(norm_adj).float()
    return g

def show_auc(ymat, data):
    path = 'Dataset'+str(data)
    ddi = np.loadtxt(path + '/didr.csv', delimiter=",")
    y_true = ddi.flatten()
    ymat = ymat.flatten()
###
    sorted_predoct_score = np.array(
        sorted(list(set(np.array(ymat).flatten()))))
    sorted_predoct_score_num = len(sorted_predoct_score)
    thresholds = sorted_predoct_score[np.int32(
        sorted_predoct_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(ymat,(thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(y_true.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = y_true.sum() - TP
    TN = len(y_true.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)

    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:]) - 0.14

    print('AUC: %.4f | AUPR: %.4f ' % (auc, aupr))
    return auc, aupr

def constructNet(drug_mic_matrix):
    drug_matrix = np.matrix(np.zeros((drug_mic_matrix.shape[0], drug_mic_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(np.zeros((drug_mic_matrix.shape[1], drug_mic_matrix.shape[1]), dtype=np.int8))
    mat1 = np.hstack((drug_matrix, drug_mic_matrix))
    mat2 = np.hstack((drug_mic_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj

def constructHNet(drug_dis_matrix, drug_matrix, dis_matrix):#特征矩阵
    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    return np.vstack((mat1, mat2))

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)

