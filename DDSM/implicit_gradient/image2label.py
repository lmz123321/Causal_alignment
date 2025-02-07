import torch
import torch.nn as nn
import torchopt
import sys
sys.path.append('/home/lijingwen/Projects')
from Counter_align.DDSM.loss.loss import i2l_multitasks_celoss


def get_implicit_i2lmodel(solver):
    '''
    img2att implicit function
    '''
    class ImplicitModel(torchopt.nn.ImplicitMetaGradientModule,
                        linear_solve=solver):
        '''
        This class is to solve the problem:  z_star(θ)=argmin_z'[CE(y',y_star)+λ(x,x')],dx_star/dθ
        '''
        def __init__(self, clser, atter, lamb, max_epochs=200, optimizer='Adam', lr=0.01):
            if max_epochs < 100:print('Warning: inner optimization max_epochs<=100, may cause strange behavior in implicit gradient estimation.')
            super().__init__()
            self.register_meta_module('atter', atter)#do implicit derivative on theta of atter
            object.__setattr__(self, 'clser', clser)#Calculate the implicit derivative of atter will use clser, but not clser
            self.lr = lr
            self.lamb = lamb
            self.max_epochs = max_epochs
            self.optimizer = optimizer

        def reset(self, z):
            self.z0 = z
            self.z = nn.Parameter(z.clone().detach_(), requires_grad=True)#optimize object:z


        def objective(self, ):
            '''
            This function is to minimize z modification while ensuring pred=clser(attr(z)) meets the specified class probability
            '''
            z_mask=self.atter(self.z)
            preds = self.clser(z_mask * self.z)
            celoss = i2l_multitasks_celoss(preds, self.target)
            norm = torch.norm(self.z - self.z0)
            return celoss + self.lamb * norm

        @torch.enable_grad()
        def solve(self, ):
            optimizer = getattr(torch.optim, self.optimizer)(params=[self.z], lr=self.lr)
            z0_mask = self.atter(self.z0)
            preds = self.clser(z0_mask * self.z0)
            mscores = [nn.Softmax(dim=1)(pred)[:, 1].unsqueeze(-1) for pred in preds]  # z0 pred：y0
            self.target = torch.cat([1 - (mscore >= 0.5).long().clone().detach() for mscore in mscores],
                                    dim=-1)  # result opposite to z0: y*

            log = {'epoch': [], 'loss': [], 'ce': [], 'norm': []}
            for epoch in range(self.max_epochs):
                ## Inner loop: Adjust z to ensure attr-generated z* meets the specified clser classification, keeping attr and clser parameters unchanged
                optimizer.zero_grad()
                loss = self.objective()
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