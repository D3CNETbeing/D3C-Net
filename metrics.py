import torch

from torch.utils.data import TensorDataset
import numpy as np

from sklearn.metrics import accuracy_score
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F


def metrc(all_labels, pred_labels,num_classes):
        acc = accuracy_score(all_labels, pred_labels)
        # Known-class ACC
        known_mask = (all_labels < num_classes)
        # print('Plabel_0',pred_labels)
        known_acc = accuracy_score(all_labels[known_mask], pred_labels[known_mask]) if known_mask.sum() > 0 else 0

        # Unknown detection AUROC
        
        # is_unknown = (all_labels == num_classes).astype(int)
        is_unknown = (all_labels >= num_classes)
        
        # indices_1 = np.where(all_labels == num_classes)[0]
        # print("INdi",indices_1)
        # print('True',set(all_labels[all_labels == num_classes]))

        # Unknown-class ACC

        unknown_preds = pred_labels[is_unknown]
        
        
        unknown_acc = np.mean(unknown_preds == num_classes) 

        # h
        hscore= 2 * known_acc * unknown_acc / (known_acc + unknown_acc)
        return acc,known_acc,unknown_acc,hscore
