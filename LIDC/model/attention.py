import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.double_conv(x)

#LIDC img2att atter Net
class LIDC_I2A_atter(nn.Module):
    def __init__(self, in_dim):
        '''
        img2att atter Net
        in_dim:C,z.shape=(B,C,H,H)
        '''
        super(LIDC_I2A_atter, self).__init__()
        self.conv = nn.Sequential(DoubleConv(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1),
                                  DoubleConv(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1))
        self.fc = nn.Conv2d(in_dim, 1, kernel_size=1, bias=False)

    def forward(self, x):
        '''
        x:(B,C,H,H)
        pred:(B,1,H,H) 0-1mask,mass mask on x
        '''
        feat = self.conv(x)
        pred= torch.sigmoid(self.fc(feat+x))
        return pred

    def print_conv_weights(self):
        print("I2A_atter conv layer weights:")
        print(self.fc.weight.data[0, 0, 0])

#LIDC att2label feat atter Net
class LIDC_A2L_feat_atter(nn.Module):
    def __init__(self, in_dim, extend_dim,out_dim,C=16):
        '''
        att2label atter
        in_dim:att categories=6*C
        extend_dim:extend_dim in fc,16/32/64...
        C:16 repeat C time begfore output to match with feat shape
        '''
        super(LIDC_A2L_feat_atter, self).__init__()
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

#LIDC img2label atter Net
class LIDC_I2L_atter(nn.Module):
    def __init__(self, in_dim):
        '''
        img2att atter Net
        in_dim:C,z.shape=(B,C,H,H)
        '''
        super(LIDC_I2L_atter, self).__init__()
        self.conv = nn.Sequential(DoubleConv(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1),
                                  DoubleConv(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1))
        self.fc = nn.Conv2d(in_dim, 1, kernel_size=1, bias=False)

    def forward(self, x):
        '''
        x:(B,C,H,H)
        pred:(B,1,H,H) 0-1mask,mass mask on x
        '''
        feat = self.conv(x)
        pred= torch.sigmoid(self.fc(feat+x))
        return pred

    def print_conv_weights(self):
        print("I2A_atter conv layer weights:")
        print(self.fc.weight.data[0, 0, 0])
