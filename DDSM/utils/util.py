import os
import json
import torch
import pandas as pd
from pathlib import Path
from collections import OrderedDict

from torch import nn


def digitize(tensor,threshold=0.5):
    '''
    digitize tensor,>0.5 set 1,<=0.5 set 0
    '''
    return (tensor>threshold).float()
def get_topk(tensor, C=16,topk=4):
    '''
    digitize tensor, tensor.shape=(B, 6), topk set 1, tensor[i]=[1, 1, 0, 1, 0, 0]
    return result,digitized tensor with topk elements set to 1.
    '''

    _, topk_indices = torch.topk(tensor, int(topk)*C, dim=1,largest=True, sorted=True)
    result = torch.zeros_like(tensor)
    result.scatter_(1, topk_indices, 1)
    return result

def minmax_normalize(tensor,dim):
    '''
    nomalize tensor into 0~1
    '''
    return (tensor-torch.amin(tensor,dim,keepdim=True))/torch.amax(tensor,dim,keepdim=True)
def find_max_number(cache_path,split):
    '''
    Find the maximum number in the pt file name in the specified directory.
    Such as under cache_path test_label_2.pt，return max_number+1=3
    '''
    max_number = -1
    for filename in os.listdir(cache_path):
        if filename.startswith(split+"_") and filename.endswith("_label.pt"):
            # Extract the number part from the file name
            parts = filename.split('_')
            number = int(parts[1])
            # update max_number
            max_number = max(max_number, number)
    return max_number+1

def att_preds_totensor(preds):
    """
    att_preds is a list,len(att_preds)=6
    att_preds[i] is a tensor,att_preds[i].shape=(B,2)
    transform whole att_preds to tensor,preds_tensor.shape=(B,6)
    """
    preds_tensor = []
    for i, pred in enumerate(preds):
        pred = torch.argmax(pred, dim=1)
        preds_tensor.append(pred.unsqueeze(1))
    preds_tensor = torch.hstack(preds_tensor).float()
    return preds_tensor

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def modify_att(att0_pred,att_pred):
    '''
    input:att0_pred shpae=(B,2),att_pred shape=(B,2),a2l_imp.att(feature) match with att_pred (attributes classification),(B,2),0~1 float prob
    output:modified_att=(B,6) modified[i][j]=1 means in the ith sample,its jth att is modified
    take jth att as example:
    att0_pred = torch.tensor([[0.2, 0.8],
                              [0.2, 0.8],
                              [0.35, 0.65],
                              [0.4, 0.6]])

    att_pred = torch.tensor([[0.25, 0.75],
                             [0.8, 0.2],
                             [0.3, 0.7],
                             [0.45, 0.55]])
    modified_att should be(1,1,0,1)
    if the value match with att0_pred argmax indice get smaller in att_pred,set the sample with 1,else set 0
    '''
    output=list()
    softmax = nn.Softmax(dim=1)
    for att_idx in range(0,6):
        argmax_indices = torch.argmax(softmax(att0_pred[att_idx]), dim=1)
        att_pred_diff = softmax(att0_pred[att_idx]) - softmax(att_pred[att_idx])
        output.append((att_pred_diff[torch.arange(att_pred_diff.size(0)), argmax_indices] > 0).int())
    return torch.stack(output,dim=1)

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self.keys=keys
        self._data = pd.DataFrame(
            index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value,n=1):
        if isinstance(value, (int, float)) is False:
            value_item=value.cpu().detach().item()
        else:
            value_item=value

        if self.writer is not None:
            if isinstance(value_item, dict):
                self.writer.add_scalars(key, value_item)

            else:
                self.writer.add_scalar(key, value_item)



        if isinstance(value_item, dict):
            for subkey in value_item.keys():
                self._data.total[subkey] += value_item[subkey] * n
                self._data.counts[subkey] += n
        else:
            self._data.total[key] += value_item * n
            self._data.counts[key] += n

        self._data.average[key] = self._data.total[key] / \
                self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)