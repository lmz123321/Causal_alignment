import torch
import torch.nn as nn
import torchopt
from Counter_align.DDSM.loss.loss import malign_celoss
def get_implicit_a2lmodel(solver):
    '''
    att2label implicit function
    '''
    class ImplicitModel(torchopt.nn.ImplicitMetaGradientModule,
                        linear_solve=solver):
        def __init__(self, clser, atter, lamb, max_epochs=200, optimizer='Adam', lr=0.01):
            if max_epochs < 100:
                print('Warning: inner optimization max_epochs<=100, may cause strange behavior in implicit gradient estimation.')
            super().__init__()
            self.register_meta_module('atter', atter)
            object.__setattr__(self, 'clser', clser)
            self.lr = lr
            self.lamb = lamb
            self.max_epochs = max_epochs
            self.optimizer = optimizer

        def reset(self,att):
            self.att0 = att
            self.att = nn.Parameter(att.clone().detach_(), requires_grad=True)

        def objective(self,):
            pred = self.clser(self.atter(self.att)*self.att)  # clser(ccce*att)
            celoss = malign_celoss(pred, self.target)
            norm = torch.norm(self.att - self.att0)
            return celoss + self.lamb * norm

        @torch.enable_grad()
        def solve(self, ):
            optimizer = getattr(torch.optim, self.optimizer)(params=[self.att], lr=self.lr)
            pred=self.clser(self.atter(self.att0)*self.att0)#binary:(64,6)->(64,2) feat:(64,6*C)->(64,2) C=16
            mscore = nn.Softmax(dim=1)(pred)[:, 1]  # att0 preds：y0
            self.target = 1 - (mscore >= 0.5).long().clone().detach()  #result opposite to att0: y*

            log = {'epoch': [], 'loss': [], 'ce': [], 'norm': []}
            for epoch in range(self.max_epochs):
                optimizer.zero_grad()
                loss = self.objective()
                norm = torch.norm(self.att - self.att0)
                celoss = loss - self.lamb * norm
                loss.backward(inputs=[self.att])
                optimizer.step()

                log['epoch'].append(epoch)
                log['loss'].append(loss.detach().cpu().item())
                log['ce'].append(celoss.detach().cpu().item())
                log['norm'].append(self.lamb * norm.detach().cpu().item())

            return log

    return ImplicitModel

