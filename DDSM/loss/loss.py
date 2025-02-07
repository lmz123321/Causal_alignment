import torch
import torch.nn as nn
from torch import Tensor
#preliminiay loss
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


#i2l_clser monitor loss
def i2l_multitasks_celoss(preds,targets):
    '''
    preds:attrbutesprediction
    targets:attribute gt
    balance:if balance samples
    '''
    loss = 0

    for i,pred in enumerate(preds):
        ce_loss = nn.CrossEntropyLoss(reduction='none')
        target = targets[:,i]
        loss += ce_loss(pred,target).mean()
    return loss/len(preds)

#i2a_clser monitor loss
#monitor loss name:"min val_i2a_celoss"
def multitasks_celoss(preds,targets,balance=True):
    '''
    preds:attrbutesprediction
    targets:attribute gt
    balance:if balance samples
    '''
    loss = 0
    if balance is True:
        sample_weights = torch.tensor([
            [4.4550, 0.5632],
            [1.3222, 0.8041],
            [0.7074, 1.7052],
            [0.6090, 2.7938],
            [0.6821, 1.8731],
            [0.6765, 1.9167]]).to(preds[0].device)#0-1 sample ratio of train set(6 categories)
    for i,pred in enumerate(preds):
        if balance is True:
            ce_loss = nn.CrossEntropyLoss(reduction='none',weight=sample_weights[i])
        else:
            ce_loss = nn.CrossEntropyLoss(reduction='none')
        target = targets[:,i]
        loss += ce_loss(pred,target).mean()
    return loss/len(preds)

def weighted_multitasks_celoss(preds,targets,ccce_weights,balance=True):
    '''
    preds:attrbutesprediction
    targets:attribute gt
    ccce_weights:ccce_score
    balance:if balance samples
    '''
    loss = 0
    if balance is True:
        sample_weights = torch.tensor([
            [4.4550, 0.5632],
            [1.3222, 0.8041],
            [0.7074, 1.7052],
            [0.6090, 2.7938],
            [0.6821, 1.8731],
            [0.6765, 1.9167]]).to(preds[0].device)#0-1 sample ratio of train set(6 categories)
    for i,pred in enumerate(preds):
        if balance is True:
            ce_loss = nn.CrossEntropyLoss(reduction='none',weight=sample_weights[i])
        else:
            ce_loss = nn.CrossEntropyLoss(reduction='none')
        target = targets[:,i]
        weight = ccce_weights[:,i]
        loss += (weight*ce_loss(pred,target)).mean()
    return loss/len(preds)


#a2l_clser monitor loss
#monitor loss name:"min val_a2l_celoss"
def malign_celoss(pred,target):
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    loss = ce_loss(pred,target).mean()
    return loss

#a2l_atter monitor loss
#monitor loss name:"min val_a2l_alignloss"
#writer losses name:'a2l_reward','a2l_punish','a2l_align_ratio','a2l_alignloss'
def inside_nod_loss(diff,mask,p=2):
    loss = torch.norm(mask*diff, p=p)
    loss = loss/diff.shape[0]
    return loss
def beyond_nod_loss(diff,mask,p=2):
    loss = torch.norm((1 - mask)*diff, p=p)
    loss = loss/diff.shape[0]
    return loss
def align_ratio(diff,mask,p=2):
    reward=inside_nod_loss(diff, mask, p=2)
    punish=beyond_nod_loss(diff,mask,p=2)
    return reward/punish
def align_loss(diff,mask,p=2,ratio=1):
    reward=inside_nod_loss(diff, mask, p=2)
    punish=beyond_nod_loss(diff,mask,p=2)
    return punish*ratio - reward



#i2a_atter monitor loss
#monitor loss name:"max val_i2a_dice"
def dice_score(pred,target):
    with torch.no_grad():
        coeff = 1 - dice_loss(pred.squeeze(), target.squeeze()).cpu()
    return coeff






