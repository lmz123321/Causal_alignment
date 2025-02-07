import torchopt
import sys
import torch
import torch.nn as nn
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.avg_pool(x)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2).unsqueeze(3)
        return x * out
class DoubleConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.adjust_channels(x)
        out = self.double_conv(x)
        return out+identity

#DDSM img2att atter Net
class DDSM_I2A_atter(nn.Module):
    def __init__(self, in_dim):
        '''
        img2att atter Net
        in_dim:C,z.shape=(B,C,H,H)
        '''
        super(DDSM_I2A_atter, self).__init__()
        self.se0 = SEBlock(in_channels=in_dim)
        self.conv= DoubleConv2(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1)
        self.conv_5x5 = DoubleConv2(in_channels=in_dim, out_channels=in_dim, kernel_size=5, padding=2)
        self.conv_7x7 =DoubleConv2(in_channels=in_dim, out_channels=in_dim, kernel_size=7, padding=3)
        self.fc = nn.Conv2d(in_dim*2, 1, kernel_size=1, bias=False)

    def forward(self, x):
        '''
        x:(B,C,H,H)
        pred:(B,1,H,H) 0-1mask,mass mask on x
        '''
        feat = self.se0(x)
        feat=self.conv(feat)
        feat_5x5 = self.conv_5x5(feat)
        feat_7x7 = self.conv_7x7(feat)
        feat = torch.cat([feat_5x5+x, feat_7x7+x], dim=1)  # 在通道维度上拼接
        pred=torch.sigmoid(self.fc(feat))
        return pred

    def print_conv_weights(self):
        print("I2A_atter conv layer weights:")
        print(self.fc.weight.data[0, 0, 0])

#DDSM img2label atter Net
class DDSM_I2L_atter(nn.Module):
    def __init__(self, in_dim):
        '''
        img2att atter Net
        in_dim:C,z.shape=(B,C,H,H)
        '''
        super(DDSM_I2L_atter, self).__init__()
        self.se0 = SEBlock(in_channels=in_dim)
        self.conv= DoubleConv2(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1)
        self.conv_5x5 = DoubleConv2(in_channels=in_dim, out_channels=in_dim, kernel_size=5, padding=2)
        self.conv_7x7 =DoubleConv2(in_channels=in_dim, out_channels=in_dim, kernel_size=7, padding=3)
        self.fc = nn.Conv2d(in_dim*2, 1, kernel_size=1, bias=False)

    def forward(self, x):
        '''
        x:(B,C,H,H)
        pred:(B,1,H,H) 0-1mask,mass mask on x
        '''
        feat = self.se0(x)
        feat=self.conv(feat)
        feat_5x5 = self.conv_5x5(feat)
        feat_7x7 = self.conv_7x7(feat)
        feat = torch.cat([feat_5x5+x, feat_7x7+x], dim=1)  # 在通道维度上拼接
        pred=torch.sigmoid(self.fc(feat))
        return pred

    def print_conv_weights(self):
        print("I2A_atter conv layer weights:")
        print(self.fc.weight.data[0, 0, 0])

#DDSM att2label atter Net
class DDSM_A2L_atter(nn.Module):
    def __init__(self, in_dim, extend_dim):
        '''
        att2label atter
        in_dim:att categories=6
        extend_dim:extend_dim in fc,16/32/64...
        '''
        super(DDSM_A2L_atter, self).__init__()
        self.fc1 = nn.Linear(in_dim, extend_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(extend_dim, in_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        x:(B,6)
        pred:(B,6) 0-1mask,predict ccce topk score on x
        '''
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        pred = self.sigmoid(x)
        return pred

    def print_conv_weights(self):
        print("A2L_atter fc layer weights:")
        print(self.fc1.weight.data[0])

#DDSM att2label feat atter Net

class DDSM_A2L_feat_atter(nn.Module):
    def __init__(self, in_dim, extend_dim,out_dim,C=16):
        '''
        att2label atter
        in_dim:att categories=6*C
        extend_dim:extend_dim in fc,16/32/64...
        C:16 repeat C time begfore output to match with feat shape
        '''
        super(DDSM_A2L_feat_atter, self).__init__()
        self.fc1 = nn.Linear(in_dim, extend_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(extend_dim, out_dim)
        self.C=C
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):#todo by ljw 20240628 feat_atter,add pred_flattened here
        '''
        x:(B,6*C)
        pred:(B,6) 0-1mask,predict ccce topk score on x
        '''
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        pred = self.sigmoid(x)#(B,6)
        pred_repeated = pred.unsqueeze(-1).repeat(1, 1, self.C)  # (B, 6, C)
        pred_flattened = pred_repeated.view(pred_repeated.size(0), -1)  # (B, 6 * C)

        return pred_flattened

    def print_conv_weights(self):
        print("A2L_atter fc layer weights:")
        print(self.fc1.weight.data[0])


