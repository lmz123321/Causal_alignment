import os
import pandas as pd
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/lijingwen/Projects')
from Counter_align.LIDC.utils.util import digitize,minmax_normalize,find_max_number
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
class LIDCpt_Data(Dataset):
    def __init__(self,cache_path,ccce_file, training, valid,shuffle,seed):
        '''
        cache_path:encoder cache_path
        ccce_file:ccce_score file
        traning:train set or not
        valid:valid set or not
        shuffle:if shuffle
        seed:shuffle seed
        '''
        if training:
            self.mode = "Train"
            split = "train"
            number = find_max_number(cache_path, split)
        else:
            self.mode = "Val" if (valid) else "Test"
            split = "val" if (valid) else "test"
            number = find_max_number(cache_path, split)
        #img
        img = torch.cat([torch.load(os.path.join(cache_path, '{}_{}_img.pt'.format(split, i))) for i in range(number)],
                      dim=0)
        #z:img latent code in encoder
        z = torch.cat([torch.load(os.path.join(cache_path, '{}_{}_z.pt'.format(split, i))) for i in range(number)],dim=0)
        self.z = (z - z.mean()) / z.std()
        #mask_gt:mass mask on z
        mask_gt = torch.cat([torch.load(os.path.join(cache_path, '{}_{}_mask.pt'.format(split, i))) for i in range(number)],dim=0)
        self.mask_gt = F.interpolate(mask_gt, size=(z.shape[-2:]), mode='bilinear')
        self.mask_gt = digitize(minmax_normalize(self.mask_gt, (2, 3)))
        #
        # for ind in range(0, 60, 10):
        #     plt.figure()
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(img[ind, 0, :])
        #     plt.imshow(mask_gt[ind, 0, :], cmap='gray', alpha=0.5)
        #     plt.title(f'{self.mode} Mask')
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(z[ind, 0, :])
        #     plt.imshow(self.mask_gt[ind, 0, :], cmap='gray', alpha=0.5)
        #     plt.title(f'{self.mode} Mask')
        #     plt.show()
        #     print("a")

        #nid:mass nodule id
        jitids = list()
        for i in range(number):
            jitids += list(torch.load(os.path.join(cache_path, '{}_{}_jitid.pt'.format(split, i))))
        nids = ['-'.join(jitid.split('-')[:-2]) for jitid in jitids]#LIDC-PID-NID-SLICE_ID-JIT_ID->LIDC-PID-NID
        #ccce topk score
        ccce_gt = pd.read_csv(ccce_file).set_index('id')
        self.ccce_gt = torch.from_numpy(ccce_gt.loc[nids].values)
        #mass nodule attributes
        self.att_gt = torch.cat([torch.load(os.path.join(cache_path, '{}_{}_label.pt'.format(split, i))) for i in range(number)], dim=0)[:,:-1]

        # malignant label
        self.y = torch.cat(
            [torch.load(os.path.join(cache_path, '{}_{}_label.pt'.format(split, i))) for i in range(number)], dim=0)[:,-1].squeeze()
        #shuffle
        if shuffle is True:
            torch.manual_seed(seed)
            shuffle_index = torch.randperm(self.y.size(0))
            self.att_gt = self.att_gt[shuffle_index]
            self.mask_gt = self.mask_gt[shuffle_index]
            self.ccce_gt = self.ccce_gt[shuffle_index]
            self.z = self.z[shuffle_index]
            self.y = self.y[shuffle_index]

    def __len__(self):
        return self.att_gt.shape[0]

    def __getitem__(self, ind):
        return self.mask_gt[ind],self.att_gt[ind],self.ccce_gt[ind],self.z[ind],self.y[ind],

class LIDCpt_DataLoader(DataLoader):
    def __init__(self,cache_path,ccce_file=None, batch_size=64,training=True, valid=False,shuffle=False,seed=122):
        self.cache_path = cache_path
        self.ccce_file=ccce_file
        if training:
            self.dataset = LIDCpt_Data(self.cache_path,self.ccce_file, training=True, valid=valid,shuffle=shuffle,seed=seed)
            super().__init__(self.dataset, batch_size, shuffle=False,num_workers=1)

        else:
            self.dataset = LIDCpt_Data(self.cache_path,self.ccce_file, training=False, valid=valid,shuffle=shuffle,seed=seed)
            super().__init__(self.dataset, len(self.dataset), shuffle=False,num_workers=1)
