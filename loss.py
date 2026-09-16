import torch

from torch.utils.data import TensorDataset
import numpy as np

from sklearn.metrics import accuracy_score
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

class Shpr_Loss(nn.Module):
    def __init__(self, known_classes, temperature=0.5):
        super().__init__()
        self.known_classes = set(known_classes)
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (N, d) 已归一化 or 未归一化的特征
        labels: (N,)
        """
        device = features.device
        
        binary_labels = torch.tensor(
            [1 if l.item() in self.known_classes else 0 for l in labels],
            device=device
        )

        
        features = F.normalize(features, dim=1)

        
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        
        N = features.size(0)
        mask = torch.eye(N, device=device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        
        logits = sim_matrix
        targets = binary_labels.unsqueeze(0).expand(N, N)  # (N, N)
        anchors = binary_labels.unsqueeze(1).expand(N, N)
        positive_mask = (targets == anchors) & (~mask)  # (N, N)

        
        log_prob = F.log_softmax(logits, dim=1)  # (N, N)

        
        mean_log_prob_pos = (positive_mask.float() * log_prob).sum(1) / (positive_mask.sum(1) + 1e-8)

        
        loss = -mean_log_prob_pos.mean()
        return loss




def orthogonality_loss(X, Y, normalize=True):
    
    if normalize:
        X = X / X.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        Y = Y / Y.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
    
    
    dot_products = torch.sum(X * Y, dim=1)  # (batch_size,)
    
    
    loss = torch.mean(dot_products ** 2)
    return loss

class SupervisedContrastiveLoss1(nn.Module):
    def __init__(self, temperature=10.0):
        """
        temperature (float): 温度系数，用于缩放相似度分布
        """
        super(SupervisedContrastiveLoss1, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        
        
        device = features.device
        batch_size = features.shape[0]
        
       
        if torch.isnan(features).any() or torch.isinf(features).any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        
        features = F.normalize(features, p=2, dim=1, eps=1e-8)
        
        
        similarity_matrix = torch.matmul(features, features.T)  # [batch_size, batch_size]
        
        
        similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
        
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).to(device)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        mask = mask & ~self_mask
        
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
       
        temp = max(self.temperature, 1e-8)
        similarity_matrix = similarity_matrix / temp
        
        
        similarity_matrix = similarity_matrix.masked_fill(self_mask, -1e20)
        log_prob = F.log_softmax(similarity_matrix, dim=1)
        
        
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        
        loss = -mean_log_prob_pos.mean()
        return loss
    
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature,sharelist):
        
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.sharelist = sharelist

    def forward(self, features, labels):
       
        
        device = features.device
       
        
        maskl = torch.isin(labels, torch.tensor(self.sharelist).to(device))
        features = features[maskl]
        labels = labels[maskl]
        batch_size = features.shape[0]
        
        if torch.isnan(features).any() or torch.isinf(features).any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        
        features = F.normalize(features, p=2, dim=1, eps=1e-8)
        
        
        similarity_matrix = torch.matmul(features, features.T)  # [batch_size, batch_size]
        
        
        similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
        
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).to(device)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        mask = mask & ~self_mask
        
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        
        temp = max(self.temperature, 1e-8)
        similarity_matrix = similarity_matrix / temp
        
        
        similarity_matrix = similarity_matrix.masked_fill(self_mask, -1e20)
        log_prob = F.log_softmax(similarity_matrix, dim=1)
        
        
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        
        loss = -mean_log_prob_pos.mean()
        return loss

class DomianKnowledgelLoss(nn.Module):
    def __init__(self, sample_ratio=0.3, mode='corr', epsilon=1e-6):
        
        super().__init__()
        self.mode = mode
        self.epsilon = epsilon
        self.sample_ratio = sample_ratio
    def _ranking_loss(self, dx, dy, margin=0.0):
        
        n = len(dx)
        if n < 2:
            return torch.tensor(0.0, device=dx.device)

        idx_i, idx_j = torch.combinations(torch.arange(n), 2).unbind(1)
        dy_diff = dy[idx_i] - dy[idx_j]
        dx_diff = dx[idx_i] - dx[idx_j]

        sign = torch.sign(dy_diff)
        ranking_loss = torch.relu(margin - sign * dx_diff)
        return torch.mean(ranking_loss) 
    def compute_freq_structure(self,y, fs=25600):
        
        Y = torch.fft.rfft(y, dim=1)
        P = torch.abs(Y) ** 2  
        freqs = torch.fft.rfftfreq(y.shape[1], 1 / fs).to(y.device)  # (T/2 + 1,)
        
        P = P + 1e-8
        P = P / P.sum(dim=1, keepdim=True)

        center = (P * freqs).sum(dim=1)  
        bandwidth = torch.sqrt(((freqs - center.unsqueeze(1))**2 * P).sum(dim=1)) 

        return center, bandwidth  

    def _random_sample_pairs(self, batch_size, labels):
        
        labels = torch.as_tensor(labels)
        unique_labels, counts = torch.unique(labels, return_counts=True)
        
        
        valid_labels = unique_labels[counts >= 2]
        if len(valid_labels) == 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        
       
        all_pairs = []
        for label in valid_labels:
            indices = torch.where(labels == label)[0]
            n = len(indices)
            
            ii, jj = torch.combinations(indices, 2, with_replacement=False).unbind(1)
            all_pairs.extend(torch.stack([ii, jj], dim=1))  
            # all_pairs.extend(torch.stack([jj, ii], dim=1))  
        
        
        num_pairs = int(batch_size**2 * self.sample_ratio)
        num_pairs = min(num_pairs, len(all_pairs))  

        
        all_pairs = torch.stack(all_pairs, dim=0) if all_pairs else torch.empty(0, 2, dtype=torch.long)
        
        
        
        if len(all_pairs) > 0:
            selected_indices = torch.randperm(len(all_pairs))[:num_pairs]
            pairs = all_pairs[selected_indices]
            
            return pairs[:, 0], pairs[:, 1]
        else:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    def _compute_sampled_dist(self, x, pairs):
        """计算采样对的距离"""
        x_i = x[pairs[0]]
        x_j = x[pairs[1]]
        return torch.norm(x_i - x_j, dim=1)

    def forward(self, x, y, label):
        
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)

        

        
        # peak, _ = torch.max(torch.abs(y), dim=1)
        # rms = torch.sqrt(torch.mean(y**2, dim=1))
        # crest_factor = peak / rms
        # crest_factor = torch.nan_to_num(crest_factor, nan=0.0, posinf=0.0, neginf=0.0)

        
        # y = torch.stack((kurtosis, crest_factor), dim=1)
        center, bandwidth = self.compute_freq_structure(y, 25600)
        y = torch.stack([center, bandwidth], dim=1)
        batch_size = x.size(0)

       
        pairs = self._random_sample_pairs(batch_size,label)
        if len(pairs[0]) == 0:
            return torch.tensor(0.0, device=x.device)

        
        dx = self._compute_sampled_dist(x, pairs)
        dy = self._compute_sampled_dist(y, pairs)

        
        if self.mode == 'mse':
            k = (torch.sum(dx * dy) / (torch.sum(dy**2) + self.epsilon)).detach()
            return nn.functional.mse_loss(dx, k * dy)
            
        elif self.mode == 'corr':
            mean_dx = torch.mean(dx)
            mean_dy = torch.mean(dy)
            cov = torch.mean((dx - mean_dx) * (dy - mean_dy))
            var_dx = torch.var(dx, unbiased=False)
            var_dy = torch.var(dy, unbiased=False)
            corr = cov * torch.rsqrt(var_dx * var_dy + self.epsilon)
            return -corr  
        elif self.mode == 'rank':
            return self._ranking_loss(dx, dy)

        else:
            raise ValueError("check the mode.")
        
class PrivateClassLoss(nn.Module):
    def __init__(self,share_list):
        super(PrivateClassLoss, self).__init__()
        self.share_list = share_list
        

    def forward(self, pred_logits, target):
        
        device = pred_logits.device
        
        mask = torch.isin(target[:, 1], torch.tensor(self.share_list).to(device))
        
        target = mask.unsqueeze(-1).float().to(device)
        
        loss = nn.BCEWithLogitsLoss()(pred_logits, target)
        
        
        return loss
    
class ShareClassLoss(nn.Module):
    def __init__(self):
        super(ShareClassLoss, self).__init__()
        
        
    def forward(self, class_logits,y,mask):
       
        if mask.sum() > 0:
            valid_logits = class_logits
            valid_labels = y[:, 1][mask]
            
            cls_loss = nn.CrossEntropyLoss()(valid_logits, valid_labels)
        else:
            cls_loss = torch.tensor(0., requires_grad=True)
        return cls_loss

class SubClassLoss(nn.Module):
    def __init__(self,num_KC):
        super(SubClassLoss, self).__init__()
        self.num_KC = num_KC
        
    def forward(self, class_logits,y,mask):
       
        mask = ~mask
        
        if mask.sum() > 0:
            valid_logits = class_logits
            valid_labels = y[:, 1][mask]-self.num_KC
            
            cls_loss = nn.CrossEntropyLoss()(valid_logits, valid_labels)
        else:
            cls_loss = torch.tensor(0., requires_grad=True)
        return cls_loss
 

class DualClassifierLossWithoutDisc(nn.Module):
    def __init__(self, shared_class_ids, alpha=1.0, gamma=1.0):
        
        super().__init__()
        self.shared_class_ids = set(shared_class_ids)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits_shared, logits_private, labels):
        
        device = logits_shared.device
        labels = labels.to(device)

        is_shared = torch.tensor([label.item() in self.shared_class_ids for label in labels], device=device)
        is_private = ~is_shared

        loss_shared_ce = (
            F.cross_entropy(logits_shared[is_shared], labels[is_shared])
            if is_shared.any() else torch.tensor(0.0, device=device)
        )

        
        # private_ids = sorted(list(set(labels[is_private].tolist())))
        # id2idx = {id_: i for i, id_ in enumerate(private_ids)}
        # private_labels = torch.tensor(
        #     [id2idx[label.item()] for label in labels[is_private]],
        #     device=device
        # )
        private_labels = labels[is_private]-len(self.shared_class_ids)
        loss_private_ce = (
            F.cross_entropy(logits_private[is_private], private_labels)
            if is_private.any() else torch.tensor(0.0, device=device)
        )

        
        probs_shared = F.softmax(logits_shared, dim=1)
        probs_private = F.softmax(logits_private, dim=1)

        
        entropy_shared_on_private = -torch.sum(
            probs_shared[is_private] * torch.log(probs_shared[is_private] + 1e-6), dim=1
        ).mean() if is_private.any() else torch.tensor(0.0, device=device)

        
        entropy_private_on_shared = -torch.sum(
            probs_private[is_shared] * torch.log(probs_private[is_shared] + 1e-6), dim=1
        ).mean() if is_shared.any() else torch.tensor(0.0, device=device)

        loss_entropy = entropy_shared_on_private + entropy_private_on_shared

        total_loss = self.gamma * (loss_shared_ce + loss_private_ce) - self.alpha * loss_entropy

        # loss_dict = {
        #     "loss_shared_ce": loss_shared_ce.item(),
        #     "loss_private_ce": loss_private_ce.item(),
        #     "loss_entropy_shared_on_private": entropy_shared_on_private.item(),
        #     "loss_entropy_private_on_shared": entropy_private_on_shared.item(),
        #     "loss_entropy_total": loss_entropy.item(),
        #     "total_loss": total_loss.item()
        # }

        return total_loss