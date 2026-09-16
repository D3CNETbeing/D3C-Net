import torch

from torch.utils.data import TensorDataset
import numpy as np

from sklearn.metrics import accuracy_score
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F


def report(mode,t,losst,Test_acc,Test_KN,Test_UNK,Test_Hscore,index={}):
    
    index['epoch'] = mode
    index['entropythre'] = t
    index['eloss'] = losst
    index['Test_ACC'] = Test_acc
    index['Test_KN'] = Test_KN
    index['Test_unk'] = Test_UNK
    index['Test_Hscore'] = Test_Hscore

    return index

def adjust_learning_rate(optimizer, base_lr, iter_num, max_iter):
    p = iter_num / max_iter
    lr = base_lr / ((1 + 10 * p) ** 0.75)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def load_known_class_dataP(npy_path, known_classes,domianid):
    """
    加载npy数据并筛选已知类别，返回标准化后的特征和标签。

    参数:
        npy_path (str): .npy文件路径，shape为 [N, 3, 3202]
        known_classes (list[int]): 已知类别列表，例如 [0,1,2,3]

    返回:
        x_tensor (torch.FloatTensor): shape [N_known, 3, 3200]
        y_tensor (torch.LongTensor): shape [N_known]
    """
    data = np.loadtxt(npy_path)  # [N, 514]
    
    
    features = data[:, :512]                # [N, 3, 3200]
    labels = data[:, -1].astype(int)       

   
    mask = np.isin(labels, known_classes)
    features = features[mask]
    labels = data[:, -2:][mask].astype(int)
    labels[:, 0] = domianid
    
    N, T = features.shape
    features_2d = features.reshape(N, -1)

    scaler = StandardScaler()
    features_2d = scaler.fit_transform(features_2d)

    # reshape  [N, 3, 3200]
    features_norm = features_2d.reshape(N,T)

    # Tensor
    x_tensor = torch.tensor(features_norm, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(x_tensor, y_tensor)

def load_target_dataP(npy_path, threshold):
    
    data = np.loadtxt(npy_path)  # [N, 3, 3202]
    
    
    features = data[:, :512]                # [N, 512]
    
    
    labels = data[:,-2:].astype(int)       # [N, 2]
    
    
    mask = labels[:, 1] >= threshold
    
    
    labels[mask, 1] = threshold
    
    
    N, T = features.shape
    features_2d = features.reshape(N, -1)

    scaler = StandardScaler()
    features_2d = scaler.fit_transform(features_2d)

    
    features_norm = features_2d.reshape(N,T)

    
    x_tensor = torch.tensor(features_norm, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(x_tensor, y_tensor)

def load_known_class_data1(npy_path, known_classes,domianid):
   
    data = np.load(npy_path)  # [N, 3, 3202]
    
    
    features = data[:, :, :3200]                # [N, 3, 3200]
    labels = data[:, 0, 3201].astype(int)       

   
    mask = np.isin(labels, known_classes)
    features = features[mask]
    labels = data[:, 0, -2:][mask].astype(int)
    labels[:, 0] = domianid
    
    N, C, T = features.shape
    features_2d = features.reshape(N, -1)

    scaler = StandardScaler()
    features_2d = scaler.fit_transform(features_2d)

    
    features_norm = features_2d.reshape(N, C, T)

    
    x_tensor = torch.tensor(features_norm, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(x_tensor, y_tensor)

def load_target_data(npy_path, threshold):
   
    data = np.load(npy_path)  # [N, 3, 3202]
    
    
    features = data[:, :, :3200]                # [N, 3, 3200]
    
    
    labels = data[:, 0, -2:].astype(int)       # [N, 2]
    
    
    mask = labels[:, 1] >= threshold
    
    
    labels[mask, 1] = threshold
    
    
    N, C, T = features.shape
    features_2d = features.reshape(N, -1)

    scaler = StandardScaler()
    features_2d = scaler.fit_transform(features_2d)

    
    features_norm = features_2d.reshape(N, C, T)

   
    x_tensor = torch.tensor(features_norm, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(x_tensor, y_tensor)

