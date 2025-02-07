import os
import cv2
import pylidc as pl
import numpy as np
from pylidc.utils import consensus
from tqdm import tqdm
from statistics import median_high
from skimage.measure import find_contours
import scipy
from matplotlib import pyplot as plt
import pandas as pd
import multiprocessing
import pandas
from functools import partial
import argparse

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def medianHigh(nodule):
    r"""
    Functs: - take a pylidc annotation class as input
            - return median high malignancy value of 4 physicians
    """
    malignancies =[]; subtleties =[]; internalstructures= []
    calcifications= []; sphericities=[]; margins =[]
    lobulations=[]; spiculations=[]; textures=[]
    for ann in nodule:
        malignancies.append(ann.malignancy); subtleties.append(ann.subtlety)
        internalstructures.append(ann.internalStructure); calcifications.append(ann.calcification)
        sphericities.append(ann.sphericity); margins.append(ann.margin)
        lobulations.append(ann.lobulation); spiculations.append(ann.spiculation); textures.append(ann.texture)

    mid_malign = median_high(malignancies); mid_sub= median_high(subtleties)
    mid_interstrc=median_high(internalstructures); mid_cal=median_high(calcifications)
    mid_spher=median_high(sphericities); mid_margin=median_high(margins)
    mid_lob=median_high(lobulations); mid_spi=median_high(spiculations); mid_text=median_high(textures)

    return mid_malign, mid_sub, mid_interstrc, mid_cal, mid_spher, mid_margin, mid_lob, mid_spi, mid_text


class LIDCData():
    def __init__(self, pid, PAD_XY, PAD_Z=10):
        self.pid = pid

        # pad controls to include how many context regions around the nodule when cutting the image
        self.PAD_XY = PAD_XY
        self.PAD_Z = PAD_Z

        self.read_volume()

    def read_volume(self, ):
        r"""
        Functs: - read volume data from dicome files
                - pre-process including HU value, re-sample, normalization, zero center
        """
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == self.pid).first()
        self.ann = scan.cluster_annotations()

        # slice_zvals stores the last coordinate of the ImagePositionPatient DICOM attribute
        thickness = scan.slice_zvals[1] - scan.slice_zvals[0]
        # load raw dicom slices, the z-order has been automatically adjusted
        slices = scan.load_all_dicom_images(verbose=False)

        # Step-1, convert to HU values
        vol = np.stack([s.pixel_array for s in slices]).astype(np.int16)
        vol[vol == -2000] = 0

        for sliceInd in range(len(slices)):

            intercept = slices[sliceInd].RescaleIntercept
            slope = slices[sliceInd].RescaleSlope

            if slope != 1:
                vol[sliceInd] = slope * vol[sliceInd].astype(np.float64)
                vol[sliceInd] = vol[sliceInd].astype(np.int16)

            vol[sliceInd] += np.int16(intercept)

        # Step-2, resampling (we can do similar things to make bbox accomedate the resampled valumns)
        new_spacing = [1, 1, 1]
        spacing = np.array([thickness] + [scan.pixel_spacing] * 2, dtype=np.float32)

        resize_factor = spacing / new_spacing
        new_real_shape = vol.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / vol.shape
        new_spacing = spacing / real_resize_factor

        vol = scipy.ndimage.zoom(vol, real_resize_factor, mode='nearest')

        # Step-3, normalization
        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0

        vol = (vol - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        vol[vol > 1] = 1.
        vol[vol < 0] = 0.

        # Step-4, zero center
        # PIXEL_MEAN = 0.25
        # vol = vol - PIXEL_MEAN

        # from z,x,y to x,y,z, in order to match pylidc
        self.vol = vol.transpose(1, 2, 0)
        self.real_resize_factor = np.roll(real_resize_factor, -1)

    def find_threeDloc(self, cmask, cbbox):
        r"""
        Functs: - find where is the nodule in the re-sampled volume, list of tuple [(sliceInd, xmin, ymin, xmax, ymax)]
                - this will be used to avoid overlapping in random jitter
        """
        threeDloc = list()
        for ind in range(cbbox[2].stop - cbbox[2].start):
            contours = find_contours(cmask[:, :, ind].astype(float), 0.5)
            # z-location in the resampled volume
            sliceind = int(np.round((ind + cbbox[2].start) * self.real_resize_factor[2]))
            if len(contours) == 0:
                continue
            else:
                nodbboxs = list()
                for contour in contours:
                    xmin = np.min(contour[:, 0]) * self.real_resize_factor[0];
                    xmax = np.max(contour[:, 0]) * self.real_resize_factor[0]
                    ymin = np.min(contour[:, 1]) * self.real_resize_factor[1];
                    ymax = np.max(contour[:, 1]) * self.real_resize_factor[1]
                    nodbboxs.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                nodbboxs = np.array(nodbboxs)
                # there maybe multiple contours detected, we choose the maximum covering box
                loc = (sliceind, nodbboxs[:, 0].min(), nodbboxs[:, 1].min(), nodbboxs[:, 2].max(), nodbboxs[:, 3].max())
            threeDloc.append(loc)

        # post processing
        # there are two reasons that sliceInd in threeDloc is not continues, i.e, 251, 255, 257, ..
        # 1. np.round induces discretion error; 2. fail to detect contours
        startSlice = threeDloc[0][0];
        stopSlice = threeDloc[-1][0] + 1
        # extra pad two slices before and after
        out = [(startSlice - 1, threeDloc[0][1], threeDloc[0][2], threeDloc[0][3], threeDloc[0][4])]

        sliceind = startSlice
        while sliceind <= stopSlice:
            isfind = False
            # where is sliceind
            for loc in threeDloc:
                if loc[0] == sliceind:
                    out.append(loc);
                    isfind = True;
                    break
            # pad previous slice
            if not isfind:
                out.append((sliceind, out[-1][1], out[-1][2], out[-1][3], out[-1][4]))
            sliceind += 1

        return out

    def patch_nod(self, verbose=False):
        r"""
        Functs: - crop each nodule from the pre-processed voulme
                - return 1. centerSlice image 2. bbox (pt1,pt1)-(pt2,pt2) of the nodule 3. malignancy score
                - if verbose, write the results into pngs for code_test purpose
        """
        outputs = {'nodind': [], 'nodimg': [], 'nodslice': [], 'nodbox': [], 'nodloc': [], 'nodmalig': [], 'nodsub': [],
                   'nodinterstr': [], 'nodcal': [],
                   'nodspher': [], 'nodmargin': [],
                   'nodlob': [], 'nodspi': [], 'nodtext': []}

        for idx, nodule in enumerate(self.ann):
            # pad arg in consensus decides to include how large context spaces around the nodule's contour
            cmask, cbbox, masks = consensus(nodule, clevel=0.5,
                                            pad=[(self.PAD_XY, self.PAD_XY), (self.PAD_XY, self.PAD_XY),
                                                 (self.PAD_Z, self.PAD_Z)])
            malignancy, subtlety, internalStructure, calcification, sphericity, margin, lobulation, spiculation, texture = medianHigh(
                nodule)

            # position in the resampled volume
            rbbox = (slice(int(np.round(cbbox[0].start * self.real_resize_factor[0])),
                           int(np.round(cbbox[0].stop * self.real_resize_factor[0])), None),
                     slice(int(np.round(cbbox[1].start * self.real_resize_factor[1])),
                           int(np.round(cbbox[1].stop * self.real_resize_factor[1])), None),
                     slice(int(np.round(cbbox[2].start * self.real_resize_factor[2])),
                           int(np.round(cbbox[2].stop * self.real_resize_factor[2])), None),)

            # three-D location of the nodule in the re-sampled image, list of tuple
            threeDloc = self.find_threeDloc(cmask, cbbox)

            # record each slice image of the nodule
            for ind, threeD in enumerate(threeDloc):
                nodslice = threeD[0];
                nodbbox = threeD[1:]
                sliceind = nodslice - rbbox[2].start
                nodvol = self.vol[rbbox][:, :, sliceind]

                # the idx-th nodule, the ind-th slice
                outputs['nodind'].append('{}-{}'.format(idx, ind))
                outputs['nodimg'].append(nodvol);
                outputs['nodslice'].append(nodslice);
                outputs['nodbox'].append(nodbbox)
                outputs['nodloc'].append(threeDloc)

                outputs['nodmalig'].append(malignancy);
                outputs['nodsub'].append(subtlety);
                outputs['nodinterstr'].append(internalStructure)
                outputs['nodcal'].append(calcification);
                outputs['nodspher'].append(sphericity);
                outputs['nodmargin'].append(margin)
                outputs['nodlob'].append(lobulation);
                outputs['nodspi'].append(spiculation);
                outputs['nodtext'].append(texture)

        return outputs


def data_process(pid, PAD_XY):
    r'''
    Funct: a helper wrapper, given a pid, save images to npy, return annotations as pd.DF
    '''
    outputs = {'filename': [], 'slice': [], 'box': [], 'loc': [], 'malignancy': [], 'subtlety': [],
               'internalStructure': [],
               'calcification': [], 'sphericity': [],
               'margin': [], 'lobulation': [], 'spiculation': [], 'texture': []}

    data = LIDCData(pid=pid, PAD_XY=PAD_XY)
    output = data.patch_nod(verbose=False)

    # save images
    for ind, nodimg in enumerate(output['nodimg']):
        saveid = output['nodind'][ind]
        filename = pid + '-' + saveid
        np.save(os.path.join(base, filename), nodimg)
        outputs['filename'].append(filename)

    # save annotations
    outputs['slice'] += output['nodslice']
    outputs['box'] += output['nodbox']
    outputs['loc'] += output['nodloc']

    outputs['malignancy'] += output['nodmalig']
    outputs['subtlety'] += output['nodsub']
    outputs['internalStructure'] += output['nodinterstr']
    outputs['calcification'] += output['nodcal']
    outputs['sphericity'] += output['nodspher']
    outputs['margin'] += output['nodmargin']
    outputs['lobulation'] += output['nodlob']
    outputs['spiculation'] += output['nodspi']
    outputs['texture'] += output['nodtext']

    outputs = pandas.DataFrame(outputs).set_index('filename')
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-p', '--padsize', default=2000, type=int, help='pad size around the nodule')
    # default pad value is much larger than the 2d image size, so pylidc.consensus will return the entire 2d image
    args = parser.parse_args()

    pids = os.listdir('/data/datasets/LIDC_IDRI/dicoms')
    base = '/data/liumingzhou/CounterAlign_output/preprocess/multiple_slices/raw/'
    os.makedirs(base, exist_ok=True)

    processor = partial(data_process, PAD_XY=args.padsize)

    p = multiprocessing.Pool(16)
    logs = list(p.map(processor, pids[:16]))
    p.close()
    p.join()

    #logs = list()
    #for pid in pids[:2]:
    #    logs.append(processor(pid))

    logs = pandas.concat(logs)
    logs.to_csv(os.path.join('/'.join(base.split('/')[:-2]),'raw.csv'.format(args.padsize)))
