import numpy as np
import os
import imageio
import pandas as pd
import random
from PIL import Image, ImageDraw
import argparse
import multiprocessing
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from time import time

import sys
sys.path.append('../')
from jsonio import load

def jitter(box, jittersize):
    r'''
    Funct: box = [xmin,ymin,xmax,ymax]
           pick a 2-d jitter randomly from (-jittersize,jittersize)
           return box+jitter
    '''
    jitx = np.random.randint(low=-jittersize, high=jittersize + 1)
    jity = np.random.randint(low=-jittersize, high=jittersize + 1)

    return [box[0] + jitx, box[1] + jity, box[2] + jitx, box[3] + jity], jitx, jity


def _iou(boxA, boxB):
    r'''
    IoU of two boxes
    '''
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def _det_overlap(annos, nid, sliceind, cropbox):
    r'''
    Funct: - check whether cropbox is legal (no overlap with other nodules)
    '''
    pid = nid.split('-')[0]
    for ind,row in annos.iterrows():
        filename = row['filename'].replace('LIDC-IDRI-','')
        _pid = filename.split('-')[0]
        _nid = '-'.join(filename.split('-')[:-1])
        # other nodules in this patient
        if pid == _pid and _nid != nid:
            # string of list to list
            locs = ast.literal_eval(row['loc'])
            for loc in locs:
                zloc = loc[0]; refbox = [loc[2], loc[1], loc[4], loc[3]]
                # in the same slice
                if zloc == sliceind and _iou(cropbox,refbox)>0:
                    return True, filename
    return False, None


def _process(row):
    r'''
    Args: - given a row in annos.csv, find jit boxes, write the cropped image, and return infos
    '''
    # logger
    output = {'filename': [], 'malignancy': [], 'box': []}
    # find which set
    pid = '-'.join(row['filename'].split('-')[:-1])
    subset = 'none'
    for _subset in ['train', 'val', 'test']:
        if pid in split[_subset]:
            subset = _subset
    if subset == 'none':
        return output  # not in train/val/test

    # attributes
    malign = row['malignancy']
    # box
    bbox = row['box']
    bbox = bbox[1:-1];
    bbox = bbox.split(',')
    xmin = int(bbox[1]);
    ymin = int(bbox[0]);
    xmax = int(bbox[3]);
    ymax = int(bbox[2])
    nodwidth = xmax - xmin;
    nodheight = ymax - ymin
    nodbox = [xmin, ymin, xmax, ymax]
    # z-location and nodule-id
    sliceind = int(row['slice'])
    # filename is LIDC-IDRI-pid-nid-sliceid e.g., nid=0123-3
    nid = '-'.join(row['filename'].replace('LIDC-IDRI-', '').split('-')[:-1])

    # load image
    npy = np.load(os.path.join(npypath, row['filename'] + '.npy'))
    img = Image.fromarray(npy)
    # find the best rand crop solution
    log = list()
    for trial in range(iterations):
        _jitbox, _jitx, _jity = jitter(nodbox, maxjit)
        _jxmin, _jymin, _jxmax, _jymax = _jitbox
        _cropbox = [_jxmin - pad, _jymin - pad, _jxmax + pad, _jymax + pad]
        # check overlap
        _isoverlap, _collider = _det_overlap(annos, nid, sliceind, _cropbox)
        # where will the lung be on the cropped image
        if not _isoverlap:
            h, w = npy.shape
            croph, cropw = _cropbox[3] - _cropbox[1], _cropbox[2] - _cropbox[0]
            cropxmin, cropymin, cropxmax, cropymax = _cropbox

            leftvert = 0 if cropxmin >= 0 else -cropxmin;
            lefthoriz = 0 if cropymin >= 0 else -cropymin
            rightvert = cropw if cropxmax <= w else w - cropxmin;
            righthoriz = croph if cropymax <= h else h - cropymin
            lungarea = (rightvert - leftvert) * (righthoriz - lefthoriz)

            log.append((lungarea, _cropbox, _jitx, _jity))
        else:
            collider = _collider

    if len(log) < numjit:
        print('Warning: inevitable overlap detected, nid: {} with {}.'.format(nid, collider))
        return output

    # the best cropboxs
    for jitind in range(numjit):
        lungarea, cropbox, jitx, jity = log[jitind]
        cropimg = img.crop(cropbox);
        cropnpy = np.array(cropimg)

        # where is the nodule on the cropped image
        nodxmin = pad - jitx;
        nodxmax = nodxmin + nodwidth
        nodymin = pad - jity;
        nodymax = nodymin + nodheight
        nodcropbox = [nodxmin, nodymin, nodxmax, nodymax]

        # save cropped image, LIDC-IDRI-pid-nid-sliceid-jitid
        np.save(os.path.join(savebase.format(subset), row['filename'] + '-{}'.format(jitind)), cropnpy)

        # visualization, about coordinate system in PIL and numpy, see CounterExp/_code_tests/2_read_npy.ipynb
        '''
        pilimg = Image.fromarray(cropnpy * 255).convert("RGB")
        drawer = ImageDraw.Draw(pilimg);
        drawer.rectangle([nodxmin - 10, nodymin - 10, nodxmax + 10, nodymax + 10], outline='red')
        pilnpy = np.array(pilimg).astype('uint8')
        imageio.imsave(os.path.join(saveimgbase.format(subset), row['filename'] + '-{}'.format(jitind) + '.jpg'),
                       pilnpy)
        '''

        # logging
        output['filename'].append(row['filename'] + '-{}'.format(jitind))
        output['box'].append(nodcropbox)
        if malign <= 3:
            output['malignancy'].append(0)
        else:
            output['malignancy'].append(1)

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rand pad and jit for multiple slices')
    parser.add_argument('-n', '--numjit', default=2, type=int,
                        help='for each nodule-slice, generate how many rand croped images')
    parser.add_argument('-p', '--padsize', default=25, type=int, help='pad size around the nodule')
    parser.add_argument('-j', '--maxjit', default=20, type=int, help='max jitter range, should be less than padsize')
    parser.add_argument('-b', '--base', default='/data/liumingzhou/CounterAlign_output/preprocess/multiple_slices',
                        help='base folder')
    parser.add_argument('-s', '--splitfile', default='../single_slice/split_mindiam7.json', help='train,val,test split file')
    parser.add_argument('-i', '--iterations', default=50, type=int, help='maximum iteration to avoid overlapping')
    args = parser.parse_args()

    numjit = args.numjit; pad = args.padsize; maxjit = args.maxjit; base = args.base; iterations = args.iterations
    splitfile = args.splitfile
    assert pad >= maxjit, 'Warning, pad less than max jitter range, can not ensure nodule is included.'
    annofile = os.path.join(base, 'raw.csv')
    npypath = os.path.join(base, 'raw')

    annos = pd.read_csv(annofile)
    split = load(splitfile)

    savebase = os.path.join(base, 'mindiam{}_pad{}_jit{}'.format(
        split['mindiam'], pad, maxjit), '{}')
    saveimgbase = os.path.join(base, 'visulization_mindiam{}_pad{}_jit{}'.format(
        split['mindiam'], pad, maxjit), '{}')
    for subset in ['train', 'val', 'test']:
        os.makedirs(savebase.format(subset), exist_ok=True)
        os.makedirs(saveimgbase.format(subset), exist_ok=True)

    rows = list()
    for index, row in annos.iterrows():
        rows.append(row)

    start = time()
    p = multiprocessing.Pool(32)
    _outputs = list(p.map(_process, rows))
    p.close()
    p.join()
    end = time()

    outputs = {'filename': [], 'malignancy': [], 'box': []}
    for output in _outputs:
        for key in outputs.keys():
            outputs[key] += output[key]

    # write results
    df = pd.DataFrame(outputs).set_index('filename')
    df.to_csv(savebase.format('annos.csv'))

    #print('Find {} benign nodules out of the {} nodules'.format(np.sum(outputs['malignancy'] == 0),
    #                                                            len(outputs['filename'])))
    print('Time cost: {:.3f} h'.format((end - start) / 3600))

