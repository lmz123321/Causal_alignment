import numpy as np
import os
import imageio
import pandas as pd
import random
from PIL import Image, ImageDraw
import argparse
import ast
from tqdm import tqdm

import sys
sys.path.append('../')
from jsonio import load


def jitter(box, jittersize, direction):
    r'''
    Funct: box = [xmin,ymin,xmax,ymax]
           pick a 2-d jitter randomly from (-jittersize,jittersize)
           return box+jitter
    '''

    jity = np.random.randint(low=-jittersize, high=jittersize + 1)

    if direction == 'left':
        jitx = np.random.randint(low=-jittersize, high=-jittersize // 2 + 1)
    elif direction == 'right':
        jitx = np.random.randint(low=jittersize // 2, high=jittersize + 1)
    elif direction == 'none':
        jitx = np.random.randint(low=-jittersize, high=jittersize + 1)
    else:
        raise ValueError('Expect direction {} to be left,right,or none'.format(direction))

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
    for ind, row in annos.iterrows():
        filename = row['filename'].replace('LIDC-IDRI-', '')
        # other nodules in this patient
        if pid in filename and filename != nid:
            # string of list to list
            locs = ast.literal_eval(row['loc'])
            for loc in locs:
                zloc = loc[0]; refbox = [loc[2], loc[1], loc[4], loc[3]]
                # in the same slice
                if zloc == sliceind and _iou(cropbox, refbox) > 0:
                    return True, filename
    return False, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='randomly jit and pad the image')
    parser.add_argument('-p', '--pad', default=25, type=int, help='pad size around the nodule')
    parser.add_argument('-j','--jit', default=20, type=int, help='maximum jit range (should be less than pad)')
    parser.add_argument('-i','--iteration',default=50, type=int, help='max iterations to find best jit')
    parser.add_argument('-b','--base',default='/data/liumingzhou/CounterAlign_output/preprocess/single_slice',help='base folder')
    parser.add_argument('-s','--splitfile',default='./split_mindiam7.json',help='train,val,test split file')
    parser.add_argument('-bias','--biastype',default='none',help='none,trainonly,or all')
    args = parser.parse_args()

    pad = args.pad
    maxjit = args.jit
    assert pad >= maxjit
    iterations = args.iteration
    base = args.base
    splitfile = args.splitfile
    biastype = args.biastype
    assert biastype in ['none', 'trainonly', 'all'], 'Bias type {} is invalid'.format(biastype)

    split = load(splitfile)
    cnt = 0
    benign_cnt = 0
    annofile = os.path.join(base, 'raw.csv')
    npypath = os.path.join(base, 'raw')
    savebase = os.path.join(base, 'mindiam{}_pad{}_jit{}_{}bias'.format(
        split['mindiam'], pad, maxjit, biastype), '{}')
    saveimgbase = os.path.join(base, 'visulization_mindiam{}_pad{}_jit{}_{}bias'.format(
        split['mindiam'], pad, maxjit, biastype), '{}')

    annos = pd.read_csv(annofile)
    outputs = {'filename': [], 'malignancy': [], 'box': []}
    for subset in ['train', 'val', 'test']:
        pids = split[subset]
        os.makedirs(savebase.format(subset), exist_ok=True)
        os.makedirs(saveimgbase.format(subset), exist_ok=True)
        for index, row in tqdm(list(annos.iterrows()), desc=subset):
            # train/val/test
            if row['filename'] not in pids:
                continue
            # attributes
            malign = row['malignancy'];subtlety = row['subtlety']
            sphericity = row['sphericity'];margin = row['margin']
            lobulation = row['lobulation'];spiculation = row['spiculation'];texture = row['texture']
            #if malign == 3:
            #    continue
            # box
            bbox = row['box']
            bbox = bbox[1:-1];bbox = bbox.split(',')
            xmin = int(bbox[1]);ymin = int(bbox[0]);xmax = int(bbox[3]);ymax = int(bbox[2])
            nodwidth = xmax - xmin;nodheight = ymax - ymin
            nodbox = [xmin, ymin, xmax, ymax]
            # z-location and nodule-id
            sliceind = int(row['slice'])
            nid = row['filename'].replace('LIDC-IDRI-', '')
            # load image
            npy = np.load(os.path.join(npypath, row['filename'] + '.npy'))
            img = Image.fromarray(npy)

            # induce a pseudo correlation bias, put benign nodules to the left, malignant ones to the right
            # todo: we can add a 0/1 on the top-left corner as bias
            direction = 'none'
            if (biastype == 'trainonly' and subset == 'train') or biastype == 'all':
                direction = 'left' if malign <= 3 else 'right'

            # find the best rand crop solution
            maxarea = 0
            for trial in range(iterations):
                _jitbox, _jitx, _jity = jitter(nodbox, maxjit, direction)
                _jxmin, _jymin, _jxmax, _jymax = _jitbox
                _cropbox = [_jxmin - pad, _jymin - pad, _jxmax + pad, _jymax + pad]
                # check overlap
                _isoverlap, _collider = _det_overlap(annos, nid, sliceind, _cropbox)

                # where will the lung be on the cropped image
                if not _isoverlap:
                    h, w = npy.shape
                    croph, cropw = _cropbox[3] - _cropbox[1], _cropbox[2] - _cropbox[0]
                    cropxmin, cropymin, cropxmax, cropymax = _cropbox

                    leftvert = 0 if cropxmin >= 0 else -cropxmin;lefthoriz = 0 if cropymin >= 0 else -cropymin
                    rightvert = cropw if cropxmax <= w else w - cropxmin;righthoriz = croph if cropymax <= h else h - cropymin
                    lungarea = (rightvert - leftvert) * (righthoriz - lefthoriz)
                    if lungarea > maxarea:
                        maxarea = lungarea
                        cropbox = _cropbox;jitx = _jitx;jity = _jity
                else:
                    collider = _collider
            if maxarea == 0:
                print('Warning: inevitable overlap detected, nid: {} with {}.'.format(nid, collider))
                continue
            # print(malign,direction,jitx)
            cropimg = img.crop(cropbox);cropnpy = np.array(cropimg)

            # where is the nodule on the cropped image
            nodxmin = pad - jitx;nodxmax = nodxmin + nodwidth
            nodymin = pad - jity;nodymax = nodymin + nodheight
            nodcropbox = [nodxmin, nodymin, nodxmax, nodymax]

            # save cropped image
            np.save(os.path.join(savebase.format(subset), row['filename']), cropnpy)

            # visualization, about coordinate system in PIL and numpy, see Counter_align/code_tests/2_read_npy.ipynb
            pilimg = Image.fromarray(cropnpy * 255).convert("RGB")
            drawer = ImageDraw.Draw(pilimg);drawer.rectangle([nodxmin - 10, nodymin - 10, nodxmax + 10, nodymax + 10], outline='red')
            drawer.text((0, 0), 'm:{}'.format(int(malign)), align="left")
            pilnpy = np.array(pilimg).astype('uint8')
            imageio.imsave(os.path.join(saveimgbase.format(subset), row['filename'] + '.png'), pilnpy)

            # counting
            cnt = cnt + 1
            outputs['filename'].append(row['filename'])
            outputs['box'].append(nodcropbox)
            if malign <= 3:
                outputs['malignancy'].append(0); benign_cnt += 1
            else:
                outputs['malignancy'].append(1)

    # write results
    df = pd.DataFrame(outputs).set_index('filename')
    df.to_csv(savebase.format('annos.csv'))

    print('Find {} benign nodules out of the {} nodules'.format(benign_cnt, cnt))
