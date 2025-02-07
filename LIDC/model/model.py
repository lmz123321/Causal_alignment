import torch
import torch.nn as nn
import torchopt
import sys
import torch
import torch.nn as nn
sys.path.append('../')
sys.path.append('/home/lijingwen/Projects/Counter_align/DDSM_hierarchical')
from hierachical_model.tools.loss import malign_celoss, multitasks_celoss
# from DDSM_hierarchical.hierachical_model.tools.util import pad_pt, steep_sigmoid, \
#     save_biggest_mask



#img2label
def get_implicit_model(solver):
    class ImplicitModel(torchopt.nn.ImplicitMetaGradientModule,
                        linear_solve=solver):
        #z_star(θ)=argmin_z'[CE(y',y_star)+λ(x,x')],dx_star/dθ
        def __init__(self, clser, atter, lamb, max_epochs=200, optimizer='Adam', lr=0.01,device=None):
            if max_epochs < 100:
                print(
                    'Warning: inner optimization max_epochs<=100, may cause strange behavior in implicit gradient estimation.')
            super().__init__()
            self.register_meta_module('atter', atter)#指定要对哪个模块的θ求隐导数
            object.__setattr__(self, 'clser', clser)#对atter求隐导数会用到clser，但不对clser求隐导数这样指定
            self.lr = lr
            self.lamb = lamb
            self.max_epochs = max_epochs
            self.optimizer = optimizer
            self.device=device
            self.atter=atter

        def reset(self, z):
            self.z0 = z
            self.z = nn.Parameter(z.clone().detach_(), requires_grad=True)#优化对象z


        def objective(self):#目标函数:z修改尽可能小，且pred=clser(attr(z))满足指定种类概率
            # z_mask = self.atter(self.z)
            pred = self.clser(self.atter(self.z) * self.z)
            celoss = malign_celoss(pred, self.target)
            norm = torch.norm(self.z - self.z0)#修改尽可能少
            return celoss + self.lamb * norm

        @torch.enable_grad()
        def solve(self, ):
            optimizer = getattr(torch.optim, self.optimizer)(params=[self.z], lr=self.lr)
            # z0_mask=self.atter(self.z0)
            # z0_save_mask=save_biggest_mask(z0_mask,self.device)#保留z0_mask最大连通块
            pred=self.clser(self.atter(self.z0)*self.z0)
            mscore=nn.Softmax(dim=1)(pred)[:, 1]#z0分类结果：y0
            self.target = 1 - (mscore >= 0.5).long().clone().detach()#与z0相反的分类结果：y*

            log = {'epoch': [], 'loss': [], 'ce': [], 'norm': []}
            for epoch in range(self.max_epochs):#内循环，在attr和clser参数不变情况下，改变z使得attr制造出的z*满足指定clser分类
                optimizer.zero_grad()
                # z_mask=self.atter(self.z)
                # self.z_save_mask = save_biggest_mask(z_mask, self.device)  # 计算z_mask最大连通块,一定要放在objective外，否则报错“Cannot access storage of Tensor”
                loss = self.objective()
                # to record inner optimization
                norm = torch.norm(self.z - self.z0)
                celoss = loss - self.lamb * norm
                loss.backward(inputs=[self.z])
                optimizer.step()

                log['epoch'].append(epoch)
                log['loss'].append(loss.detach().cpu().item())
                log['ce'].append(celoss.detach().cpu().item())
                log['norm'].append(self.lamb * norm.detach().cpu().item())

            return log

    return ImplicitModel
# #img2label save_biggest_mask
# def get_implicit_model_biggest(solver):
#     class ImplicitModel(torchopt.nn.ImplicitMetaGradientModule,
#                         linear_solve=solver):
#         #z_star(θ)=argmin_z'[CE(y',y_star)+λ(x,x')],dx_star/dθ
#         def __init__(self, clser, atter, lamb, max_epochs=200, optimizer='Adam', lr=0.01,device=None):
#             if max_epochs < 100:
#                 print(
#                     'Warning: inner optimization max_epochs<=100, may cause strange behavior in implicit gradient estimation.')
#             super().__init__()
#             self.register_meta_module('atter', atter)#指定要对哪个模块的θ求隐导数
#             object.__setattr__(self, 'clser', clser)#对atter求隐导数会用到clser，但不对clser求隐导数这样指定
#             self.lr = lr
#             self.lamb = lamb
#             self.max_epochs = max_epochs
#             self.optimizer = optimizer
#             self.device=device
#             self.atter=atter
#
#         def reset(self, z):
#             self.z0 = z
#             self.z = nn.Parameter(z.clone().detach_(), requires_grad=True)#优化对象z
#
#
#         def objective(self):#目标函数:z修改尽可能小，且pred=clser(attr(z))满足指定种类概率
#             z_mask = self.atter(self.z)
#             pred = self.clser(self.z_save_mask*z_mask * self.z)
#             celoss = malign_celoss(pred, self.target)
#             norm = torch.norm(self.z - self.z0)#修改尽可能少
#             return celoss + self.lamb * norm
#
#         @torch.enable_grad()
#         def solve(self, ):
#             optimizer = getattr(torch.optim, self.optimizer)(params=[self.z], lr=self.lr)
#             z0_mask=self.atter(self.z0)
#             z0_save_mask=save_biggest_mask(z0_mask,self.device)#保留z0_mask最大连通块
#             pred=self.clser(z0_save_mask*self.atter(self.z0)*self.z0)
#             mscore=nn.Softmax(dim=1)(pred)[:, 1]#z0分类结果：y0
#             self.target = 1 - (mscore >= 0.5).long().clone().detach()#与z0相反的分类结果：y*
#
#             log = {'epoch': [], 'loss': [], 'ce': [], 'norm': []}
#             for epoch in range(self.max_epochs):#内循环，在attr和clser参数不变情况下，改变z使得attr制造出的z*满足指定clser分类
#                 optimizer.zero_grad()
#                 z_mask=self.atter(self.z)
#                 self.z_save_mask = save_biggest_mask(z_mask, self.device)  # 计算z_mask最大连通块,一定要放在objective外，否则报错“Cannot access storage of Tensor”
#                 loss = self.objective()
#                 # to record inner optimization
#                 norm = torch.norm(self.z - self.z0)
#                 celoss = loss - self.lamb * norm
#                 loss.backward(inputs=[self.z])
#                 optimizer.step()
#
#                 log['epoch'].append(epoch)
#                 log['loss'].append(loss.detach().cpu().item())
#                 log['ce'].append(celoss.detach().cpu().item())
#                 log['norm'].append(self.lamb * norm.detach().cpu().item())
#
#             return log
#
#     return ImplicitModel
#DDSM img2att
def get_implicit_attmodel(solver):
    class ImplicitModel(torchopt.nn.ImplicitMetaGradientModule,
                        linear_solve=solver):
        #z_star(θ)=argmin_z'[CE(y',y_star)+λ(x,x')],dx_star/dθ
        def __init__(self, clser, atter, lamb, max_epochs=200, optimizer='Adam', lr=0.01):
            if max_epochs < 100:
                print(
                    'Warning: inner optimization max_epochs<=100, may cause strange behavior in implicit gradient estimation.')
            super().__init__()
            self.register_meta_module('atter', atter)#指定要对哪个模块的θ求隐导数
            object.__setattr__(self, 'clser', clser)#对atter求隐导数会用到clser，但不对clser求隐导数这样指定
            self.lr = lr
            self.lamb = lamb
            self.max_epochs = max_epochs
            self.optimizer = optimizer

        def reset(self, z, w):
            self.z0 = z
            self.z = nn.Parameter(z.clone().detach_(), requires_grad=True)#优化对象z
            self.w = w  # causal score

        def objective(self, ):#目标函数:z修改尽可能小，且pred=clser(attr(z))满足指定种类概率
            z_mask=self.atter(self.z)
            # preds = self.clser(self.z_save_mask*z_mask * self.z)
            preds = self.clser(z_mask * self.z)
            celoss = multitasks_celoss(preds, self.target)
            norm = torch.norm(self.z - self.z0)#修改尽可能少
            return celoss + self.lamb * norm

        @torch.enable_grad()
        def solve(self, ):
            optimizer = getattr(torch.optim, self.optimizer)(params=[self.z], lr=self.lr)
            z0_mask = self.atter(self.z0)
            # z0_save_mask = save_biggest_mask(z0_mask, self.z0.device)  # 保留z0_mask最大连通块
            # preds = self.clser(z0_save_mask * z0_mask * self.z0)
            preds = self.clser(z0_mask * self.z0)
            mscores = [nn.Softmax(dim=1)(pred)[:, 1].unsqueeze(-1) for pred in preds]#z0分类结果：y0
            self.target = torch.cat([1 - (mscore >= 0.5).long().clone().detach() for mscore in mscores], dim=-1)#与z0相反的分类结果：y*

            log = {'epoch': [], 'loss': [], 'ce': [], 'norm': []}
            for epoch in range(self.max_epochs):#内循环，在attr和clser参数不变情况下，改变z使得attr制造出的z*满足指定clser分类
                optimizer.zero_grad()
                z_mask = self.atter(self.z)
                # self.z_save_mask = save_biggest_mask(z_mask, self.z.device)
                loss = self.objective()
                # to record inner optimization
                norm = torch.norm(self.z - self.z0)
                celoss = loss - self.lamb * norm
                loss.backward(inputs=[self.z])
                optimizer.step()

                log['epoch'].append(epoch)
                log['loss'].append(loss.detach().cpu().item())
                log['ce'].append(celoss.detach().cpu().item())
                log['norm'].append(self.lamb * norm.detach().cpu().item())
            return log

    return ImplicitModel

#DDSM att2label
def get_implicit_a2lmodel(solver):
    class ImplicitModel(torchopt.nn.ImplicitMetaGradientModule,
                        linear_solve=solver):
        def __init__(self, clser, atter, lamb, max_epochs=200, optimizer='Adam', lr=0.01):
            if max_epochs < 100:
                print(
                    'Warning: inner optimization max_epochs<=100, may cause strange behavior in implicit gradient estimation.')
            super().__init__()
            self.register_meta_module('atter', atter)
            object.__setattr__(self, 'clser', clser)
            self.lr = lr
            self.lamb = lamb
            self.max_epochs = max_epochs
            self.optimizer = optimizer

        def reset(self,pred_att):
            self.pred_att0 = pred_att
            self.pred_att = nn.Parameter(pred_att.clone().detach_(), requires_grad=True)

        def objective(self, ):
            pred = self.clser(self.atter(self.pred_att)*self.pred_att)  # clser(w*feat)
            celoss = malign_celoss(pred, self.target)
            norm = torch.norm(self.pred_att - self.pred_att0)
            return celoss + self.lamb * norm

        @torch.enable_grad()
        def solve(self, ):
            optimizer = getattr(torch.optim, self.optimizer)(params=[self.pred_att], lr=self.lr)
            pred=self.clser(self.atter(self.pred_att0)*self.pred_att0)#(128,6)->(128,2)
            mscore = nn.Softmax(dim=1)(pred)[:, 1]  # z0分类结果：y0
            self.target = 1 - (mscore >= 0.5).long().clone().detach()  # 与z0相反的分类结果：y*

            log = {'epoch': [], 'loss': [], 'ce': [], 'norm': []}
            for epoch in range(self.max_epochs):
                optimizer.zero_grad()
                loss = self.objective()
                # to record inner optimization
                norm = torch.norm(self.pred_att - self.pred_att0)
                celoss = loss - self.lamb * norm
                loss.backward(inputs=[self.pred_att])
                optimizer.step()

                log['epoch'].append(epoch)
                log['loss'].append(loss.detach().cpu().item())
                log['ce'].append(celoss.detach().cpu().item())
                log['norm'].append(self.lamb * norm.detach().cpu().item())
            return log

    return ImplicitModel

# def get_implicit_model(solver):
#     class ImplicitModel(torchopt.nn.ImplicitMetaGradientModule,
#                         linear_solve=solver):
#         def __init__(self, clser, atter, lamb, max_epochs=200, optimizer='Adam', lr=0.01):
#             if max_epochs < 100:
#                 print(
#                     'Warning: inner optimization max_epochs<=100, may cause strange behavior in implicit gradient estimation.')
#             super().__init__()
#             self.register_meta_module('atter', atter)
#             object.__setattr__(self, 'clser', clser)
#             self.lr = lr
#             self.lamb = lamb
#             self.max_epochs = max_epochs
#             self.optimizer = optimizer
#
#         def reset(self, feat,feat_binary):
#             self.feat=feat
#             self.feat_binary0 = feat_binary
#             self.feat_binary = nn.Parameter(feat_binary.clone().detach_(), requires_grad=True)
#
#         def objective(self, ):
#             weight_feat_binary = self.atter(self.feat_binary).unsqueeze(dim=2) * self.feat#(128,6,1)*(128,6,64)
#             weight_feat_binary_ = weight_feat_binary.reshape(weight_feat_binary.size(0), -1) # (128,6*64)
#             pred = self.clser(weight_feat_binary_)  # clser(w*feat)
#             celoss = malign_celoss(pred, self.target)
#             norm = torch.norm(self.feat_binary - self.feat_binary0)
#             return celoss + self.lamb * norm
#
#         @torch.enable_grad()
#         def solve(self, ):
#             optimizer = getattr(torch.optim, self.optimizer)(params=[self.feat_binary], lr=self.lr)
#             self.pred_w=self.atter(self.feat_binary0)#pred_w:(128,6)
#             weight_feat_binary0 = self.pred_w.unsqueeze(dim=2) * self.feat#(128,6,1)*(128,6,64)
#             weight_feat_binary0_ = weight_feat_binary0.reshape(weight_feat_binary0.size(0), -1)  # (128,6*64)
#             mscore = self.clser(weight_feat_binary0_)[:, 1].unsqueeze(-1)
#             self.target = 1 - (mscore >= 0.5).long().clone().detach().squeeze()
#
#             log = {'epoch': [], 'loss': [], 'ce': [], 'norm': []}
#             for epoch in range(self.max_epochs):
#                 optimizer.zero_grad()
#                 loss = self.objective()
#                 # to record inner optimization
#                 norm = torch.norm(self.feat_binary - self.feat_binary0)
#                 celoss = loss - self.lamb * norm
#                 loss.backward(inputs=[self.feat_binary])
#                 optimizer.step()
#
#                 log['epoch'].append(epoch)
#                 log['loss'].append(loss.detach().cpu().item())
#                 log['ce'].append(celoss.detach().cpu().item())
#                 log['norm'].append(self.lamb * norm.detach().cpu().item())
#             return log
#
#     return ImplicitModel

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
            nn.ReLU(),  # 去掉 inplace=True 参数
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())  # 去掉 inplace=True 参数

        self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.adjust_channels(x)
        out = self.double_conv(x)
        return out+identity
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

#LIDC atterNet
class LIDC_atterNet(nn.Module):
    def __init__(self, in_dim, reduction=None):
        super(LIDC_atterNet, self).__init__()
        self.conv = nn.Sequential(DoubleConv(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1),
                                  DoubleConv(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1))
        self.fc = nn.Conv2d(in_dim, 1, kernel_size=1, bias=False)

    def forward(self, x):
        feat = self.conv(x)
        return torch.sigmoid(self.fc(feat + x))

class DDSM_atterNet(nn.Module):
    def __init__(self, in_dim, reduction=None):
        super(DDSM_atterNet, self).__init__()
        self.se0 = SEBlock(in_channels=in_dim)
        self.conv= nn.Sequential(DoubleConv2(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1))
        self.conv_5x5 = DoubleConv2(in_channels=in_dim, out_channels=in_dim, kernel_size=5, padding=2)
        self.conv_7x7 =DoubleConv2(in_channels=in_dim, out_channels=in_dim, kernel_size=7, padding=3)
        self.fc = nn.Conv2d(in_dim*2, 1, kernel_size=1, bias=False)

    def forward(self, x):
        feat = self.se0(x)
        feat=self.conv(feat)
        feat_5x5 = self.conv_5x5(feat)
        feat_7x7 = self.conv_7x7(feat)
        feat = torch.cat([feat_5x5+x, feat_7x7+x], dim=1)  # 在通道维度上拼接
        return torch.sigmoid(self.fc(feat))
class DDSM_atterJITNet(nn.Module):
    def __init__(self, in_dim, reduction=None):
        super(DDSM_atterJITNet, self).__init__()
        self.se0 = SEBlock(in_channels=in_dim, reduction_ratio=8)
        self.conv = nn.Sequential(
            # DoubleConv2(in_channels=in_dim, out_channels=in_dim, kernel_size=1, padding=0),
            DoubleConv2(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1),
        )
        # self.se1=SEBlock(in_channels=in_dim)
        self.conv_3x3 = DoubleConv2(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1)
        self.conv_5x5 = DoubleConv2(in_channels=in_dim, out_channels=in_dim, kernel_size=5, padding=2)
        self.conv_7x7 = DoubleConv2(in_channels=in_dim, out_channels=in_dim, kernel_size=7, padding=3)

        self.fc = nn.Conv2d(in_dim * 3, 1, kernel_size=1, bias=False)

    def forward(self, x):
        feat = self.se0(x)
        feat = self.conv(feat)
        feat_3x3 = self.conv_3x3(feat)
        feat_5x5 = self.conv_5x5(feat)
        feat_7x7 = self.conv_7x7(feat)
        feat = torch.cat([feat_5x5 + x, feat_7x7 + x, feat_3x3 + x], dim=1)  # 在通道维度上拼接
        feat = torch.sigmoid(self.fc(feat))

        return feat

#LIDC image2label Net
class LIDC_i2lNet(nn.Module):
    def __init__(self, in_dim):
        super(LIDC_i2lNet, self).__init__()
        self.feat_dim = in_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_channels=self.feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.feat_dim), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim), nn.ReLU(inplace=True),
                                nn.Linear(self.feat_dim, 2))

    def forward(self, x):
        feat = self.pool(self.conv(x)).squeeze()
        return self.fc(feat)

#DDSM image2label Net
class DDSM_i2lNet(nn.Module):
    def __init__(self, in_dim):
        super(DDSM_i2lNet, self).__init__()
        self.feat_dim = in_dim
        self.conv0=nn.Sequential(
            nn.Conv2d(in_dim, out_channels=self.feat_dim*2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.feat_dim*2), nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.feat_dim*2, out_channels=self.feat_dim*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.feat_dim*2), nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.feat_dim*2, out_channels=self.feat_dim*2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.feat_dim*2), nn.ReLU(),
        )
        self.spatial_attention = SpatialAttention()  # 添加 SpatialAttention
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(self.feat_dim*4, self.feat_dim*2),nn.Linear(self.feat_dim*2, 2))

        # 调用模型参数初始化方法
        self.init_weights()

    def init_weights(self):
        # 对模型参数进行随机初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):

        x0 = self.conv0(x)
        x_3 = self.conv3(x0)
        x_3 = self.spatial_attention(x_3)
        x_5 = self.conv5(x0)
        x_5 = self.spatial_attention(x_5)
        feat = torch.cat([x_3 + x0, x_5 + x0], dim=1)  # 在通道维度上拼接
        feat = self.pool(feat).squeeze()
        return self.fc(feat)

#LIDC image2latt Net
class LIDC_i2aNet(nn.Module):
    def __init__(self, in_dim, expansion, num_tasks):
        super(LIDC_i2aNet, self).__init__()
        self.feat_dim = in_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_channels=self.feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.feat_dim), nn.ReLU())

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fcs = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.feat_dim, expansion * self.feat_dim), nn.ReLU(inplace=True),
                           nn.Linear(expansion * self.feat_dim, 2)) for _ in range(num_tasks)])

    def forward(self, x):
        x = self.conv(x)
        feat = self.pool(x).squeeze()

        pred = list()
        for fc in self.fcs:
            pred.append(fc(feat))
        return pred

    # by ljw
    # extract feature from fc layer
    def forward_feature(self, x):
        x = self.conv(x)
        feat = self.pool(x).squeeze()

        pred_feat = list()
        for fc in self.fcs:
            pred_feat.append(fc[0](feat))
        feat_stack = torch.stack(pred_feat, dim=0)
        return feat_stack

    def forward_feature_binary(self, x):
        x = self.conv(x)
        feat = self.pool(x).squeeze()

        pred_feats_binary = list()
        pred_feats = list()
        for fc in self.fcs:
            pred_feat = fc[0](feat)
            pred_feats.append(pred_feat)
            pred_feats_binary.append(fc[2](fc[1](pred_feat)))
        feat_stack = torch.stack(pred_feats, dim=0)
        feat_binary_stack = torch.stack(pred_feats_binary, dim=0)
        return feat_stack, feat_binary_stack

#DDSM image2att Net
class DDSM_i2aNet(nn.Module):
    def __init__(self, in_dim):
        super(DDSM_i2aNet, self).__init__()
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
        self.spatial_attention = SpatialAttention()  # 添加 SpatialAttention
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(self.feat_dim*4, self.feat_dim*2),nn.Linear(self.feat_dim*2, 2))
        self.fcs = nn.ModuleList(
            [nn.Sequential(
                # nn.Linear(self.feat_dim*4, self.feat_dim*2),
                nn.Linear(self.feat_dim*2, 2)
            ) for _ in range(6)])

        # 调用模型参数初始化方法
        self.init_weights()

    def init_weights(self):
        # 对模型参数进行随机初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):

        x0 = self.conv0(x)
        x_3 = self.conv3(x0)
        x_3 = self.spatial_attention(x_3)
        x_5 = self.conv5(x0)
        x_5 = self.spatial_attention(x_5)
        feat = torch.cat([x_3 + x0, x_5 + x0], dim=1)  # 在通道维度上拼接
        feat = self.pool(feat).squeeze()

        pred = list()
        for fc in self.fcs:
            pred.append(fc(feat))
        return pred



#DDSM image2label Net
class DDSM_JITi2lNet(nn.Module):
    def __init__(self, in_dim):
        super(DDSM_JITi2lNet, self).__init__()
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
        self.spatial_attention = SpatialAttention(kernel_size=5)  # 添加 SpatialAttention
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(self.feat_dim*2, 2))

        # 调用模型参数初始化方法
        self.init_weights()

    def init_weights(self):
        # 对模型参数进行随机初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):

        x = self.conv0(x)
        x_3 = self.conv3(x)
        x_3 = self.spatial_attention(x_3)
        x_5 = self.conv5(x)
        x_5 = self.spatial_attention(x_5)
        feat = torch.cat([x_3 + x, x_5 + x], dim=1)  # 在通道维度上拼接
        feat = self.pool(feat).squeeze()
        return self.fc(feat)




#LIDC att2label Net
class LIDC_a2lNet(nn.Module):
    def __init__(self, input_size,dim,output_size):#input_size(128,6)
        super(LIDC_a2lNet, self).__init__()

        self.flatten=nn.Flatten()
        self.fc1 = nn.Linear(input_size, dim)#dim=64
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x=self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)#output:(128,6)
        return x

#DDSM att2label cls Net
class DDSM_a2l_clser(nn.Module):
    def __init__(self, in_dim):#input_dim
        super(DDSM_a2l_clser, self).__init__()
        self.in_dim=in_dim
        self.model = nn.Sequential(
            nn.Linear(in_dim,2),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self,feat):#feat (6,)
        pred = self.model(feat)
        return pred

#DDSM att2label atter Net
class DDSM_a2l_atter(nn.Module):
    def __init__(self, input_size, dim):  # input_size(128,6)
        super(DDSM_a2l_atter, self).__init__()

        self.fc1 = nn.Linear(input_size, dim)  # dim=64
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(dim, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # output:(128,6)
        return x