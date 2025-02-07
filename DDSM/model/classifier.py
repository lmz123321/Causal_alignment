import torch
import torch.nn as nn
from torch.nn import init


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(1), nn.ReLU(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out*x

#DDSM image2att Net
class DDSM_I2A_clser(nn.Module):
    def __init__(self, in_dim):
        '''
        img2att clser Net
        in_dim:C,z.shape=(B,C,H,H)
        '''
        super(DDSM_I2A_clser, self).__init__()
        self.feat_dim = in_dim
        self.conv0=nn.Sequential(
            nn.Conv2d(in_dim, out_channels=self.feat_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.feat_dim), nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.feat_dim, out_channels=self.feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.feat_dim), nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.feat_dim, out_channels=self.feat_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.feat_dim), nn.ReLU(),
        )
        self.spatial_attention = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fcs = nn.ModuleList([nn.Linear(self.feat_dim*2, 2) for _ in range(6)])

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        '''
        x:x.shape=(B,C,H,H)
        pred:attributes prediction,pred is a list,len(pred)=6,pred[i].shape=(B,2)
        '''
        x0 = self.conv0(x)
        x_3 = self.conv3(x0)
        x_3 = self.spatial_attention(x_3)
        x_5 = self.conv5(x0)
        x_5 = self.spatial_attention(x_5)
        feat = torch.cat([x_3 + x0, x_5 + x0], dim=1)
        feat = self.pool(feat).squeeze()

        pred = list()
        for fc in self.fcs:
            pred.append(fc(feat))
        return pred

#DDSM image2att feat Net
class DDSM_I2A_feat_clser(nn.Module):
    def __init__(self, in_dim):
        '''
        img2att clser Net
        in_dim:C,z.shape=(B,C,H,H)
        '''
        super(DDSM_I2A_feat_clser, self).__init__()
        self.feat_dim = in_dim
        self.conv0=nn.Sequential(
            nn.Conv2d(in_dim, out_channels=self.feat_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.feat_dim), nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.feat_dim, out_channels=self.feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.feat_dim), nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.feat_dim, out_channels=self.feat_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.feat_dim), nn.ReLU(),
        )
        self.spatial_attention = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fcs = nn.ModuleList([nn.Sequential(nn.Linear(self.feat_dim*2, self.feat_dim),
                                                nn.ReLU(inplace=True),nn.Linear(self.feat_dim,2))
                                  for _ in range(6)])#todo by ljw 20240625 add feat fc here

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        '''
        x:x.shape=(B,C,H,H)
        pred:attributes prediction,pred is a list,len(pred)=6,pred[i].shape=(B,2)
        '''
        x0 = self.conv0(x)
        x_3 = self.conv3(x0)
        x_3 = self.spatial_attention(x_3)
        x_5 = self.conv5(x0)
        x_5 = self.spatial_attention(x_5)
        feat = torch.cat([x_3 + x0, x_5 + x0], dim=1)
        feat = self.pool(feat).squeeze()

        pred = list()

        for fc in self.fcs:
            pred.append(fc(feat))
        return pred

    def forward_feature(self, x):
        '''
        x:x.shape=(B,C,H,H)
        feat_stack:attributes feature stack,feat_stack is a tensor,feat_stack.shape=(B,6,C)
        '''
        x0 = self.conv0(x)
        x_3 = self.conv3(x0)
        x_3 = self.spatial_attention(x_3)
        x_5 = self.conv5(x0)
        x_5 = self.spatial_attention(x_5)
        feat = torch.cat([x_3 + x0, x_5 + x0], dim=1)
        feat = self.pool(feat).squeeze()

        pred_feat = list()
        for fc in self.fcs:
            pred_feat.append(fc[0](feat))
        feat_stack = torch.stack(pred_feat, dim=0)#(6,B,C)
        feat_stack=feat_stack.permute(1, 0, 2)#(B,6,C)
        return feat_stack

    def forward_feat2att(self, feat_stack):
        '''
        feat_stack:attributes feature stack,feat_stack is a tensor,feat_stack.shape=(B,6*C)
        pred:attributes prediction,pred is a list,len(pred)=6,pred[i].shape=(B,2)
        '''
        feat_stack_ = feat_stack.reshape(feat_stack.size(0),6,-1).permute(1, 0, 2)#(B,6*C)->(B,6,C)->(6,B,C)

        pred = list()
        for i in range(0,6):
            tmp=self.fcs[i][2](self.fcs[i][1](feat_stack_[i]))
            pred.append(tmp)

        return pred

#DDSM image2label feat Net
class DDSM_I2L_feat_clser(nn.Module):
    def __init__(self, in_dim):
        '''
        img2att clser Net
        in_dim:C,z.shape=(B,C,H,H)
        '''
        super(DDSM_I2L_feat_clser, self).__init__()
        self.feat_dim = in_dim
        self.conv0=nn.Sequential(
            nn.Conv2d(in_dim, out_channels=self.feat_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.feat_dim), nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.feat_dim, out_channels=self.feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.feat_dim), nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.feat_dim, out_channels=self.feat_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.feat_dim), nn.ReLU(),
        )
        self.spatial_attention = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fcs = nn.ModuleList([nn.Sequential(nn.Linear(self.feat_dim*2, self.feat_dim),
                                                nn.ReLU(inplace=True),nn.Linear(self.feat_dim,2))
                                  for _ in range(1)])#todo by ljw 20240625 add feat fc here

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        '''
        x:x.shape=(B,C,H,H)
        pred:malignancy prediction,pred is a list,len(pred)=6,pred[i].shape=(1,2)
        '''
        x0 = self.conv0(x)
        x_3 = self.conv3(x0)
        x_3 = self.spatial_attention(x_3)
        x_5 = self.conv5(x0)
        x_5 = self.spatial_attention(x_5)
        feat = torch.cat([x_3 + x0, x_5 + x0], dim=1)
        feat = self.pool(feat).squeeze()

        pred = list()

        for fc in self.fcs:
            pred.append(fc(feat))
        return pred

    def forward_feature(self, x):
        '''
        x:x.shape=(B,C,H,H)
        feat_stack:malignancy feature stack,feat_stack is a tensor,feat_stack.shape=(B,1,C)
        '''
        x0 = self.conv0(x)
        x_3 = self.conv3(x0)
        x_3 = self.spatial_attention(x_3)
        x_5 = self.conv5(x0)
        x_5 = self.spatial_attention(x_5)
        feat = torch.cat([x_3 + x0, x_5 + x0], dim=1)
        feat = self.pool(feat).squeeze()

        pred_feat = list()
        for fc in self.fcs:
            pred_feat.append(fc[0](feat))
        feat_stack = torch.stack(pred_feat, dim=0)#(1,B,C)
        feat_stack=feat_stack.permute(1, 0, 2)#(B,1,C)
        return feat_stack

    def forward_feat2att(self, feat_stack):
        '''
        feat_stack:attributes feature stack,feat_stack is a tensor,feat_stack.shape=(B,1*C)
        pred:malignancy prediction,pred is a list,len(pred)=1,pred[i].shape=(B,2)
        '''
        feat_stack_ = feat_stack.reshape(feat_stack.size(0),1,-1).permute(1, 0, 2)#(B,1*C)->(B,1,C)->(1,B,C)

        pred = list()
        for i in range(0,1):
            tmp=self.fcs[i][2](self.fcs[i][1](feat_stack_[i]))
            pred.append(tmp)

        return pred

#DDSM att2label cls Net
class DDSM_A2L_clser(nn.Module):
    def __init__(self, in_dim):
        '''
        att2label clser
        in_dim:att categories=6
        '''
        super(DDSM_A2L_clser, self).__init__()
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
