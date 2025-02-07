import os
import pandas as pd
import numpy as np

import sys
sys.path.append('../')
from jsonio import save
import argparse

def box2diameter(box):
    '''
    box = string([ymin,xmin,ymax,xmax])
    diameter:= max(height,width)
    '''
    box = box[1:-1]
    box = box.split(',')
    xmin = int(box[1]); ymin = int(box[0]); xmax = int(box[3]); ymax = int(box[2])
    width = xmax - xmin; height = ymax - ymin
    return max(width,height)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split train,val,test')
    parser.add_argument('-d', '--mindiam', default=7, type=int, help='minimum nodule diameter')
    parser.add_argument('-r','--ratio',default=[3,1,1], type=int, help='train,val,test ratio')
    parser.add_argument('-b','--base',default='/data/liumingzhou/CounterAlign_output/preprocess/single_slice',
                        type=str,help='path to the base folder')
    args = parser.parse_args()

    mindiam = args.mindiam
    ratio = args.ratio
    base = args.base
    seed = 123

    np.random.seed(seed)
    ratio = np.array(ratio) / sum(ratio)

    file = os.path.join(base, 'raw.csv')
    df = pd.read_csv(file)

    boxs = df['box'].tolist()
    diameters = list(map(box2diameter, boxs))
    nids = df['filename'].tolist()
    touse = list()
    for nid, diameter in zip(nids, diameters):
        if diameter >= mindiam:
            touse.append(nid)

    nid2pid = lambda string: string.split('-')[-2]

    touse_pid = list(set(list(map(nid2pid, touse))))
    num = len(touse_pid)
    np.random.shuffle(touse_pid)

    # split by patients
    trainpid = touse_pid[:int(num * ratio[0])]
    valpid = touse_pid[int(num * ratio[0]):int(num * (ratio[0] + ratio[1]))]
    testpid = touse_pid[int(num * (ratio[0] + ratio[1])):]

    train = [nid for nid in touse if nid2pid(nid) in trainpid]
    val = [nid for nid in touse if nid2pid(nid) in valpid]
    test = [nid for nid in touse if nid2pid(nid) in testpid]

    assert set(map(nid2pid,train)).intersection(set(map(nid2pid,test)))==set()

    split = {'mindiam': mindiam, 'seed': seed,
             'ratio': ratio.tolist(),
             'number': [len(train), len(val), len(test)],
             'train': train, 'val': val, 'test': test}
    save(split, 'split_mindiam{}.json'.format(mindiam, ratio))



