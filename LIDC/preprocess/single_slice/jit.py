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


def jitter(box, jittersize):
    r'''
    Funct: box = [xmin,ymin,xmax,ymax]
           pick a 2-d jitter randomly from (-jittersize,jittersize)
           return box+jitter
    '''

    jity = np.random.randint(low=-jittersize, high=jittersize + 1)
    jitx = np.random.randint(low=-jittersize, high=jittersize + 1)

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
    parser.add_argument('-b','--base',default='/data/liumingzhou/CounterAlign_output/preprocess/single_slice',type=str,help='base folder')
    parser.add_argument('-s','--splitfile',default='./split_mindiam7.json',type=str,help='train,val,test split file')
    parser.add_argument('-bias','--biastype',default='none',type=str,help='none, trainonly, or all')
    parser.add_argument('-oracle', '--oracle',default=False,type=bool,
                        help='if True, will mask all non-nodule regions into zero to create an oracle dataset')
    args = parser.parse_args()

    pad = args.pad
    maxjit = args.jit
    assert pad >= maxjit
    iterations = args.iteration
    base = args.base
    splitfile = args.splitfile
    biastype = args.biastype
    assert biastype in ['none', 'trainonly', 'all'], 'Bias type {} is invalid'.format(biastype)
    # how many pad you want to add in align_loss definition
    mask_pad = 5

    split = load(splitfile)
    cnt = 0
    benign_cnt = 0
    annofile = os.path.join(base, 'raw.csv')
    npypath = os.path.join(base, 'raw')
    savebase = os.path.join(base, 'mindiam{}_pad{}_jit{}_{}bias_anti_collision_{}oracle'.format(
        split['mindiam'], pad, maxjit, biastype, args.oracle), '{}')
    saveimgbase = os.path.join(base, 'visulization_mindiam{}_pad{}_jit{}_{}bias_anti_collision_{}oracle'.format(
        split['mindiam'], pad, maxjit, biastype, args.oracle), '{}')

    annos = pd.read_csv(annofile)
    outputs = {'filename': [], 'malignancy': [], 'box': []}
    for subset in ['train','val', 'test']:
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

            # find the best rand crop solution
            maxarea = 0
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
            cropimg = img.crop(cropbox)
            # where is the nodule on the cropped image
            nodxmin = pad - jitx; nodxmax = nodxmin + nodwidth
            nodymin = pad - jity; nodymax = nodymin + nodheight
            nodcropbox = [nodxmin, nodymin, nodxmax, nodymax]

            # add a + on the top-right corner for malignant nodule, - for benign ones
            flag1 = biastype=='trainonly' and subset=='train'
            flag2 = biastype=='all'
            if flag1 or flag2:
                symbol = '-' if malign <= 3 else '+'
            else:
                symbol = random.choice(['+','-'])
            # where to add the symbol, we avoid the +/- symbol overlap with the nodule (pad5) mask
            positions = [(5, 0), (5, cropimg.size[1] - 10), (cropimg.size[0] - 10, 0), (cropimg.size[0] - 10, cropimg.size[1] - 10)]
            densities = [np.array(cropimg)[position[1] + 5, position[0] + 2] for position in positions]
            nod_box = [max(0, nodymin - mask_pad), max(0, nodxmin - mask_pad),
                       min(cropimg.size[1] - 1, nodymax + mask_pad), min(cropimg.size[0] - 1, nodxmax + mask_pad)]
            for position,density in zip(positions,densities):
                color = 0 if density > 0.5 else 1
                sym_x, sym_y = position[1] + 5, position[0] + 2
                sym_box = [sym_x-2,sym_y-2,sym_x+2,sym_y+2]
                iou = _iou(sym_box,nod_box)
                if iou == 0:
                    break
            assert iou==0, 'Fail to find property way to add the +/- symbol.'
            bias_drawer = ImageDraw.Draw(cropimg)
            bias_drawer.text(position, symbol, align="center",fill=color)
            cropnpy = np.array(cropimg)

            if args.oracle:
                mask = np.zeros_like(cropnpy)
                mask[max(0,nodymin-5):min(cropnpy.shape[0]-1,nodymax+5),
                     max(0,nodxmin-5):min(cropnpy.shape[1]-1,nodxmax+5)] = 1
                cropnpy = cropnpy * mask

            # save cropped image
            np.save(os.path.join(savebase.format(subset), row['filename']), cropnpy)

            # visualization, about coordinate system in PIL and numpy, see Counter_align/code_tests/2_read_npy.ipynb
            pilimg = Image.fromarray(cropnpy * 255).convert("RGB")
            drawer = ImageDraw.Draw(pilimg);drawer.rectangle([nodxmin - mask_pad, nodymin - mask_pad, nodxmax + mask_pad, nodymax + mask_pad], outline='red')
            drawer.text((cropimg.size[0]//2, cropimg.size[1]-10), 'm:{}'.format(int(malign)), align="center")
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
