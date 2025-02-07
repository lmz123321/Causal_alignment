from torch import Tensor
import torch
import torch.nn as nn
#preliminary metrics


def multitasks_accuracy(preds,targets):
    accs = list()
    for i,pred in enumerate(preds):
        target = targets[:,i]
        accs.append(accuracy(pred,target))
    return accs
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

def binary_acc(output, target):#output:(128,6) target:(128,6)
    output_tensor = (output > 0.5).float()
    correct = torch.sum(output_tensor == target, dim=0).float()/output.shape[0]
    return correct

#i2a_clser metrics
#writer metric name:'i2a_avg_acc'
def multitasks_avg_accuracy(preds,targets):
    accs=multitasks_accuracy(preds,targets)
    avg_acc=sum(accs) / len(accs)
    return avg_acc

#i2a_atter metrics
#writer metric name:'i2a_avg_acc', 'i2a_celoss'
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

def precision(outputs,target):
    with torch.no_grad():
        correct = (outputs.long() & target.long()).sum()
        precision = correct / outputs.sum()
    return precision.cpu()

def recall(outputs,target):
    with torch.no_grad():
        correct = (outputs.long() & target.long()).sum()
        recall = correct / target.sum()
    return recall.cpu()

def dice_score(pred,target):
    with torch.no_grad():
        coeff = 1 - dice_loss(pred.squeeze(), target.squeeze()).cpu()
    return coeff

#a2l_clser_metrics
#writer metric name:'i2a_avg_acc'
def accuracy(output, target):
    pred = torch.argmax(output, dim=1)
    correct = torch.sum(pred == target).item()
    return correct / len(target)


#a2l_atter metrics
#writer metric name:'ccce_acc'
def ccce_acc(output, target):#output:(128,6) target:(128,6)
    accs=binary_acc(output, target)
    atter_acc = sum(accs) / len(accs)
    return atter_acc.item()
















