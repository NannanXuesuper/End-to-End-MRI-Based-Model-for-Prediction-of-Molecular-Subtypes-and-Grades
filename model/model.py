import torch.nn as nn
import torch
import warnings
import os
import numpy as np
import time
import torch.nn.functional as F
import warnings
from torch.autograd import Variable

warnings.filterwarnings('ignore')
class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c


class GatedAttention(nn.Module):
    def __init__(self, feat,relu):
        super(GatedAttention, self).__init__()
        self.L = feat//2
        self.D = feat//4
        self.K = 1
        self.feat = feat
        self.relu = relu
        
        if self.relu:
            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(self.feat, self.L),
                nn.ReLU(),
            )
        else:

            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(self.feat, self.L)
            )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier2 = nn.Sequential(
            nn.Linear(self.L*self.K, 1)
        )

        self.classifier1 = nn.Sequential(
            nn.Linear(self.feat, 2)
        )
        
    def forward(self, x):
        
        H = self.feature_extractor_part2(x)  # NxL
        # H = x
        # print("AAAAAAAAAA:",H.shape)
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL
        
        Y_prob_bag = self.classifier2(M)
        # Y_hat =torch.max(Y_prob, dim=1)[1]
        # Y_prob_ins = self.classifier1(x)
        return Y_prob_bag
class BilinearFusion(nn.Module):
    def __init__(self, skip=0, use_bilinear=True, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25,relu = True):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0
        if relu:
            self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
            self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
            self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

            self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
            self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
            self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

            self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
            self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU())
            self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU())
        else:
            self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1))
            self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
            self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1),  nn.Dropout(p=dropout_rate))

            self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2))
            self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
            self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.Dropout(p=dropout_rate))

            self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
            self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256))
            self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid))
        #init_max_weights(self)

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        # o1 = torch.cat((o1, torch.Tensor(o1.shape[0], 1).fill_(1)), 1)
        # o2 = torch.cat((o2, torch.Tensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out
class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 512, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        # print(a.shape)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
import torch.nn.functional as F               
class fusionmodel(nn.Module):
    def __init__(self,n_classes= 1,fusion_layres = "concat",dropout=0.25,scale_dim1=8,gate_path=1,scale_dim2=8,gate_omic=1,skip=True,relu = True,top_k = 200,use_bilinear = 1 ):
        super(fusionmodel, self).__init__()
        # self.model2 = swin_tiny_patch4_window7_224()
        # self.model1 = resnet50()
        
        self.fusion = fusion_layres
        self.size_dict_resnet =  [top_k,top_k//2,top_k//4 ]

        self.n_classes = n_classes
        if relu:
            fc = [ nn.Linear(self.size_dict_resnet[0], self.size_dict_resnet[1]), nn.ReLU(), nn.Dropout(dropout),]
            self.rho1 = nn.Sequential(*[nn.Linear(self.size_dict_resnet[1], self.size_dict_resnet[2]), nn.ReLU(), nn.Dropout(dropout)])
            self.rho2 = nn.Sequential(*[nn.Linear(self.size_dict_resnet[1], self.size_dict_resnet[2]), nn.ReLU(), nn.Dropout(dropout)])
            self.rho3 = nn.Sequential(*[nn.Linear(self.size_dict_resnet[1], self.size_dict_resnet[2]), nn.ReLU(), nn.Dropout(dropout)])
          
        else:
            fc = [nn.Linear(self.size_dict_resnet[0], self.size_dict_resnet[1]), nn.Dropout(dropout)]
            self.rho1 = nn.Sequential(*[nn.Linear(self.size_dict_resnet[1], self.size_dict_resnet[2]),nn.Dropout(dropout)])
            self.rho2 = nn.Sequential(*[nn.Linear(self.size_dict_resnet[1], self.size_dict_resnet[2]),  nn.Dropout(dropout)])
            self.rho3 = nn.Sequential(*[nn.Linear(self.size_dict_resnet[1], self.size_dict_resnet[2]),  nn.Dropout(dropout)])
            
        attention_net = Attn_Net_Gated(L=self.size_dict_resnet[1], D=self.size_dict_resnet[1], dropout=0.25, n_classes=1)
        fc.append(attention_net)  
        self.feat1 = nn.Sequential(*fc)
        

       
          
        self.feat2 = nn.Sequential(*fc)
        self.feat3 = nn.Sequential(*fc)
        

        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(self.size_dict_resnet[2]*3,  self.size_dict_resnet[2])])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=self.size_dict_resnet[2], dim2=self.size_dict_resnet[2], scale_dim1=scale_dim1, gate1=gate_path, scale_dim2=scale_dim2, gate2=gate_omic, skip=skip, mmhid=self.size_dict_resnet[2],relu= relu,use_bilinear=use_bilinear)
        elif self.fusion == 'lrb':
            self.mm = LRBilinearFusion(dim1=512, dim2=512, scale_dim1=scale_dim1, gate1=gate_path, scale_dim2=scale_dim2, gate2=gate_omic)
        else:
            self.mm = None
        self.classifier_mm = nn.Linear(self.size_dict_resnet[2], n_classes)


    def forward(self, feature1,feature2,feature3):
    
        A,feature1 = self.feat1(feature1)
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        feature1 = torch.mm(A, feature1)
        feature1 = self.rho1(feature1)

        A, feature2 = self.feat2(feature2)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        feature2 = torch.mm(A, feature2)
        feature2 = self.rho2(feature2)

        A, feature3 = self.feat3(feature3)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        feature3 = torch.mm(A, feature3)
        feature3 = self.rho3(feature3)

        if self.fusion == 'bilinear':
            h_mm = self.mm(feature1, feature2)
        elif self.fusion == 'concat':
            h_mm = self.mm(torch.cat([feature1, feature2,feature3], axis=1))
        elif self.fusion == 'lrb':
            h_mm  = self.mm(feature1, feature2) # logits needs to be a [1 x 4] vector 
            return h_mm
        # print(h_mm.size())
        logits  = self.classifier_mm(h_mm)
        # hazards = torch.sigmoid(logits)
        # S = torch.cumprod(1 - hazards, dim=1)


        return logits