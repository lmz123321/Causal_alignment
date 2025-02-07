import torch
import torch.nn as nn
from torch.nn import init

#LIDC att2label cls Net
class LIDC_A2L_clser(nn.Module):
    def __init__(self, in_dim):
        '''
        att2label clser
        in_dim:att categories=6
        '''
        super(LIDC_A2L_clser, self).__init__()
        self.in_dim=in_dim
        self.model = nn.Sequential(
            nn.Linear(in_dim,2),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        '''
        x:(B,6)
        output:(B,6)
        '''
        output = self.model(x)
        return output

    def reset_parameters(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.Softmax):
                # ReLU and Softmax do not have parameters to initialize
                continue


#LIDC img2att feat Net
class LIDC_I2A_feat_clser(nn.Module):
    def __init__(self, in_dim):
        super(LIDC_I2A_feat_clser, self).__init__()
        self.feat_dim = in_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_channels=self.feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.feat_dim), nn.ReLU())

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fcs = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.feat_dim,self.feat_dim), nn.ReLU(inplace=True),
                           nn.Linear(self.feat_dim, 2)) for _ in range(6)])

    def forward(self, x):
        '''
        x:x.shape=(B,C,H,H)
        pred:attributes prediction,pred is a list,len(pred)=6,pred[i].shape=(B,2)
        '''
        x = self.conv(x)
        feat = self.pool(x).squeeze()

        pred = list()
        for fc in self.fcs:
            pred.append(fc(feat))
        return pred


    def forward_feature(self, x):
        '''
        x:x.shape=(B,C,H,H)
        feat_stack:attributes feature stack,feat_stack is a tensor,feat_stack.shape=(B,6,C)
        '''
        x = self.conv(x)
        feat = self.pool(x).squeeze()
        pred_feat = list()
        for fc in self.fcs:
            pred_feat.append(fc[0](feat))
        feat_stack = torch.stack(pred_feat, dim=0)#(6,B,C)
        feat_stack = feat_stack.permute(1, 0, 2) # (B,6,C)
        return feat_stack

    def forward_feat2att(self, feat_stack):
        '''
        feat_stack:attributes feature stack,feat_stack is a tensor,feat_stack.shape=(B,6*C)
        pred:attributes prediction,pred is a list,len(pred)=6,pred[i].shape=(B,2)
        '''
        feat_stack_ = feat_stack.reshape(feat_stack.size(0), 6, -1).permute(1, 0, 2)  # (B,6*C)->(B,6,C)->(6,B,C)

        pred = list()
        for i in range(0, 6):
            tmp = self.fcs[i][2](self.fcs[i][1](feat_stack_[i]))
            pred.append(tmp)

        return pred

#LIDC img2att feat Net
class LIDC_I2L_feat_clser(nn.Module):
    def __init__(self, in_dim):
        super(LIDC_I2L_feat_clser, self).__init__()
        self.feat_dim = in_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_channels=self.feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.feat_dim), nn.ReLU())

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fcs = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.feat_dim,self.feat_dim), nn.ReLU(inplace=True),
                           nn.Linear(self.feat_dim, 2)) for _ in range(1)])

    def forward(self, x):
        '''
        x:x.shape=(B,C,H,H)
        pred:attributes prediction,pred is a list,len(pred)=6,pred[i].shape=(B,2)
        '''
        x = self.conv(x)
        feat = self.pool(x).squeeze()

        pred = list()
        for fc in self.fcs:
            pred.append(fc(feat))
        return pred


    def forward_feature(self, x):
        '''
        x:x.shape=(B,C,H,H)
        feat_stack:attributes feature stack,feat_stack is a tensor,feat_stack.shape=(B,1,C)
        '''
        x = self.conv(x)
        feat = self.pool(x).squeeze()
        pred_feat = list()
        for fc in self.fcs:
            pred_feat.append(fc[0](feat))
        feat_stack = torch.stack(pred_feat, dim=0)#(1,B,C)
        feat_stack = feat_stack.permute(1, 0, 2) # (B,1,C)
        return feat_stack

    def forward_feat2att(self, feat_stack):
        '''
        feat_stack:attributes feature stack,feat_stack is a tensor,feat_stack.shape=(B,1*C)
        pred:attributes prediction,pred is a list,len(pred)=1,pred[i].shape=(B,2)
        '''
        feat_stack_ = feat_stack.reshape(feat_stack.size(0), 1, -1).permute(1, 0, 2)  # (B,1*C)->(B,1,C)->(1,B,C)

        pred = list()
        for i in range(0, 1):
            tmp = self.fcs[i][2](self.fcs[i][1](feat_stack_[i]))
            pred.append(tmp)

        return pred
