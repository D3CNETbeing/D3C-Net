import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import  DataLoader,ConcatDataset
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from utils import *
from model import *
from loss import *
from metrics import *
import pandas as pd
import os

class CNNWithBN(nn.Module):
    def __init__(self,out_dim,fdim,fdim2, num_SC,num_PC):
        super(CNNWithBN, self).__init__()
        
        self.conv1 = TinyResNet1D(out_dim =out_dim)
        # self.mapc = nn.Linear(out_dim, 64)
        # self.mapd = nn.Linear(out_dim, 64)
       
        self.mapc = nn.Sequential(
            nn.Linear(out_dim, fdim)
            
        )
        self.mapd = nn.Sequential(
            nn.Linear(out_dim, fdim)
           
        )
        # self.classfier = nn.Linear(64, num_SC+num_PC)
        self.dimc = nn.Linear(fdim, 3)

        self.cross_domian_sample = Cross_domain_sampling(fdim,fdim,8,8,"NAVG")
        # self.rawclassfier = nn.Linear(fdim, num_SC+num_PC)

        self.sharemap = nn.Sequential(
            nn.Linear(fdim, fdim2),
            nn.BatchNorm1d(fdim2),
            nn.LeakyReLU(),
            nn.Linear(fdim2, fdim2)  
        )
        #nn.Linear(fdim, fdim2)
        self.privatemap = nn.Sequential(
            nn.Linear(fdim, fdim2),
            nn.BatchNorm1d(fdim2),
            nn.LeakyReLU(),
            nn.Linear(fdim2, fdim2)  
        )
        #nn.Linear(fdim, fdim2)
        
        
        self.shclassfier = nn.Linear(fdim2, num_SC+1)
        self.prclassfier = nn.Linear(fdim2, num_PC+1)
        self.shpr = nn.Linear(fdim+8, 1)#1
        
        self.grl = GRL(lambd=1.0)
        self.G = nn.Sequential(
            # nn.Linear(fdim2, 2*fdim2),
            # nn.LeakyReLU(),
            # nn.Linear(2*fdim2, fdim2)
            nn.Linear(fdim2, fdim2)
        )
        # self.D = nn.Sequential(
        #     nn.Linear(fdim, fdim2),
        #     nn.BatchNorm1d(fdim2),
        #     nn.LeakyReLU()
        # )
        self.Dsh = nn.Sequential(
            nn.Linear(fdim2, 1)
        )
        self.Dpr = nn.Sequential(
            nn.Linear(fdim2, 1)
        )

        

    def forward(self, x,mask,mode):
        x = self.conv1(x)  # → [B, 3, 3200, 1]
        cf = self.mapc(x)
        df = self.mapd(x)
        
        dl = self.dimc(df)

        conf = self.cross_domian_sample(cf,df)
        # confl = self.rawclassfier(conf)

        shf = self.sharemap(cf)
        
        
        prf = self.privatemap(cf)
        

        # shprfeat = self.shprmap(conf)
        shprf = self.shpr(conf)
        if mode=='train':
       
            if mask.any():
                gpr = self.G(shf[mask].detach())
                
                shcl = self.shclassfier(shf[mask])
                
            else:
                gpr = torch.randn_like(prf)
                shcl = torch.randn_like(shf)

            noise_pr = self.prclassfier(gpr)    
            

            if (~mask).any():
                gsh = self.G(prf[~mask].detach())
                
                prcl = self.prclassfier(prf[~mask])
                
            else:
                gsh = torch.randn_like(shf)
                prcl = torch.randn_like(prf)


            noise_sh = self.shclassfier(gsh)
            
            

            pr_grl = self.prclassfier(self.grl(gpr))
            sh_grl = self.shclassfier(self.grl(gsh))
            # Dsh = self.D(torch.cat([noise_sh, shf], dim=0))
            # Dpr = self.D(torch.cat([noise_pr, prf], dim=0))

            # dshf = self.Dsh(torch.cat([noise_sh, shf], dim=0))
            # dprf = self.Dpr(torch.cat([noise_pr, prf], dim=0))
            
            
            return cf,df,dl,conf,shcl,prcl,shprf,noise_sh, noise_pr,sh_grl,pr_grl,shf,prf,gpr,gsh
        if mode=='test':
            shcl = self.shclassfier(shf)
            prcl = self.prclassfier(prf)
            return cf,df,dl,conf,shcl,prcl,shprf

  



