import torch

from torch.utils.data import TensorDataset
import numpy as np

from sklearn.metrics import accuracy_score
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = F.relu(out)
        return out

class TinyResNet1D(nn.Module):
    def __init__(self, input_channels=1, out_dim=512, dropout=0.2):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=128, stride=2, padding=2), #128
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.out_dim = out_dim
        
        self.layer1 = self._make_layer(16, 16, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(16, 32, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(32, 32, stride=2, dropout=dropout)
        
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, self.out_dim)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, stride, dropout):
        return nn.Sequential(
            ResidualBlock1D(in_channels, out_channels, stride, dropout)
            # ResidualBlock1D(out_channels, out_channels, stride=1, dropout=dropout)
        )
    
    def forward(self, x):
        x = self.initial(x[:,0:1,:]) # x shape: (batch, channels, length) Piston reshape
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.adaptive_pool(x)  # (batch, channels, 1)
        x = x.view(x.size(0), -1)  
        x = self.fc(x)             
        return x


from torch.autograd import Function
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class GRL(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return GradReverse.apply(x, self.lambd)
    

class Cross_domain_sampling(nn.Module):
    """class feature与任意domainfeature结合"""
    def __init__(self, class_feadim,domain_feadim,pooldim,out_dim,pool):
        super().__init__()

        self.kernel_size = int(domain_feadim/pooldim) ###pooldim必须被doamin_feadim整除

        if pool == "AVG":
            self.Pooling =nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.kernel_size)
        elif pool == "NAVG":
            self.Pooling =nn.Sequential(
            # nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.kernel_size),
            nn.Linear(domain_feadim, out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim)  
        )
            

        # self.contrast_projection = nn.Sequential(
        #     nn.Linear(class_feadim+pooldim, out_dim,bias=False),
        #     nn.BatchNorm1d(out_dim, affine=False)

        # )
             
    
    def forward(self, class_feat,domain_feat):
        
        # domain_feat = self.Pooling(domain_feat.unsqueeze(1)).squeeze(dim=1).detach()
        domain_feat = self.Pooling(domain_feat).squeeze(dim=1).detach()
        domain_feat = F.normalize(domain_feat, p=2, dim=1, eps=1e-8)
        shuffled_indices = torch.randperm(domain_feat.size(0))
        shuffled_domian = domain_feat[shuffled_indices]
        class_feat = torch.cat((class_feat,shuffled_domian),dim=1)
        # class_feat = self.contrast_projection(class_feat)
        

        return class_feat
 