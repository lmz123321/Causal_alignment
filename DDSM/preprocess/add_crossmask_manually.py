import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from matplotlib import pyplot as plt
import random
import pandas as pd
from tqdm.notebook import tqdm

import sys
sys.path.append('../../causal_align/utils/')
sys.path.append('/home/lijingwen/Projects/Counter_align_old/baseline/breast_cancer_diagnosis-master/src')
from jit import cal_mask_boundaries

sys.path.append('../../causal_align/utils/')


#在指定坐标的图像上绘制一个方块。
def draw(img,box):
    ymin,xmin,ymax,xmax = box
    img[xmin,ymin]=1
    img[xmax,ymax]=1
    img[xmax,ymin]=1
    img[xmin,ymax]=1

def get_iou(boxA, boxB):
    '''
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


def cross_mark(npy, coord, symbol, size=4, thickness=1, fill_value=1):
    '''
    add a +/- mark at coord; size controls the size of the mark
    '''
    x, y = coord
    npy[x - thickness + 1:x + thickness, y - size + 1:y + size] = fill_value
    if symbol == '+':
        npy[x - size + 1:x + size, y - thickness + 1:y + thickness] = fill_value
def cube_mark(npy, coord,  size=4,fill_value=1):
    '''
    add a +/- mark at coord; size controls the size of the mark
    '''
    x, y = coord
    # npy[x - thickness + 1:x + thickness, y - size + 1:y + size] = fill_value
    # if symbol == '+':
    #     npy[x - size + 1:x + size, y - thickness + 1:y + thickness] = fill_value
    npy[x - size + 1:x + size, y - size + 1:y + size] = fill_value
def add_cross(row,data_dir,picsize):
    if not os.path.exists(os.path.join(data_dir,f"BIG-JIT{str(picsize)}-CROSS")):
        os.mkdir(os.path.join(data_dir,f"BIG-JIT{str(picsize)}-CROSS"))

    npy = np.load(os.path.join(data_dir,f"BIG-JIT{str(picsize)}",f"{row['id']}.npy"))
    malign = row['pathology']
    pad = 4
    size = 6
    thickness = 2

    x_min,x_max,y_min,y_max=row['jmask_xmin'],row['jmask_xmax'],row['jmask_ymin'],row['jmask_ymax']
    nod_box = [max(0, y_min - pad), max(0, x_min - pad), min(y_max + pad, npy.shape[1] - 1),
               min(x_max + pad, npy.shape[0] - 1)]

    # set the symbol according to label
    if data_dir.split('/')[-1] == 'Train':
        symbol = '+' if malign == 1 else '-'
    else:
        symbol = '-' if malign == 1 else '+'

    # choose the best corner
    positions = [(8,8), (8, npy.shape[0] -8), (npy.shape[1] -8,8),
                 (npy.shape[1] - 8, npy.shape[0] -8)]
    densities = [npy[pos[1], pos[0]] for pos in positions]

    for position, density in zip(positions, densities):
        sym_x, sym_y = position[1], position[0]
        sym_box = [sym_y - size + 1, sym_x - size + 1, sym_y + size, sym_x + size]
        iou = get_iou(sym_box, nod_box)  # 判断是否与nod_box交并

        # by ljw判断周围像素值和+-差异是否达到15以上
        region_indices = np.s_[sym_x - size + 1:sym_x + size, sym_y - size + 1:sym_y + size]  # 指定区域
        tmp=npy[region_indices]
        # if_clear = np.all(npy[region_indices] < (250 / 255))
        # if iou == 0 and if_clear:
        #     break
        if iou == 0:
            break

    # assert iou == 0, 'Fail to find property way to add the +/- symbol.'
    cross_mark(npy, (sym_x, sym_y), thickness=thickness, symbol=symbol, size=size, fill_value=255)
    if iou!=0:
        plt.figure(figsize=(3, 3))
        plt.imshow(npy, cmap='gray')
        plt.title(str(malign))

        plt.axhline(y_min, color='red')
        plt.axhline(y_max, color='red')
        plt.axvline(x_min, color='red')
        plt.axvline(x_max, color='red')
        plt.show()
        print("a")

    # assert if_clear, 'Fail to find clear corner.'


    np.save(os.path.join(data_dir,f"BIG-JIT{str(picsize)}-CROSS",f"{row['id']}.npy"), npy)



    # id=row['id']
    # dest_filepath = os.path.join(data_dir, 'UNRESIZE', f'{id}.npy')
    # img_mask_out_path = os.path.join(data_dir, 'UNRESIZE-MASK', f'{id}.npy')
    # jit_img_savepath = os.path.join(data_dir, 'BIG-JIT', f'{id}.npy')
    # jit_mask_savepath = os.path.join(data_dir, 'BIG-JIT-MASK', f'{id}.npy')
    # img = np.load(dest_filepath)
    # mask = np.load(img_mask_out_path)
    # jit_img = np.load(jit_img_savepath)
    # jit_mask = np.load(jit_mask_savepath)
    # # 创建画布和子图
    # fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    # # 第一张子图
    # axs[0].imshow(img)
    # # axs[0].imshow(mask, alpha=0.5)  # 绘制透明mask
    # axs[0].axvline(x=row['jit_xmin'], color='r', linestyle='--', linewidth=2)  # 绘制jit坐标线
    # axs[0].axvline(x=row['jit_xmax'], color='r', linestyle='--', linewidth=2)
    # axs[0].axhline(y=row['jit_ymin'], color='r', linestyle='--', linewidth=2)
    # axs[0].axhline(y=row['jit_ymax'], color='r', linestyle='--', linewidth=2)
    # xmin, ymin, xmax, ymax = cal_mask_boundaries(mask)
    # axs[0].axvline(x=xmin, color='g', linestyle='-', linewidth=2)  # 绘制jit坐标线
    # axs[0].axvline(x=xmax, color='g', linestyle='-', linewidth=2)
    # axs[0].axhline(y=ymin, color='g', linestyle='-', linewidth=2)
    # axs[0].axhline(y=ymax, color='g', linestyle='-', linewidth=2)
    #
    # axs[0].set_title('Original Image with Mask and Jit Box')
    #
    # # 第二张子图
    # axs[1].imshow(jit_img)
    # axs[1].imshow(jit_mask, alpha=0.5)  # 绘制透明jit_mask
    # # axs[1].axvline(x=row['jmask_xmin'], color='r', linestyle='--', linewidth=2)  # 绘制jit_mask坐标线
    # # axs[1].axvline(x=row['jmask_xmax'], color='r', linestyle='--', linewidth=2)
    # # axs[1].axhline(y=row['jmask_ymin'], color='r', linestyle='--', linewidth=2)
    # # axs[1].axhline(y=row['jmask_ymax'], color='r', linestyle='--', linewidth=2)
    # axs[1].set_title('Jit Image with Jit Mask')
    #
    #
    # axs[2].imshow(npy, cmap='gray')
    # axs[2].set_title(str(malign))
    #
    # axs[2].axhline(y_min, color='red')
    # axs[2].axhline(y_max, color='red')
    # axs[2].axvline(x_min, color='red')
    # axs[2].axvline(x_max, color='red')
    # plt.show()
    #
    # print("a")
def add_cube(row,data_dir,picsize):
    if not os.path.exists(os.path.join(data_dir,f"BIG-JIT{str(picsize)}-CUBE")):
        os.mkdir(os.path.join(data_dir,f"BIG-JIT{str(picsize)}-CUBE"))

    npy = np.load(os.path.join(data_dir,f"BIG-JIT{str(picsize)}",f"{row['id']}.npy"))
    malign = row['pathology']
    pad = 4
    size = 6
    thickness = 2

    x_min,x_max,y_min,y_max=row['jmask_xmin'],row['jmask_xmax'],row['jmask_ymin'],row['jmask_ymax']
    nod_box = [max(0, y_min - pad), max(0, x_min - pad), min(y_max + pad, npy.shape[1] - 1),
               min(x_max + pad, npy.shape[0] - 1)]

    # set the symbol according to label
    if data_dir.split('/')[-1] == 'Train':
        symbol = '+' if malign == 1 else '-'
    else:
        symbol = '-' if malign == 1 else '+'

    # choose the best corner
    positions = [(8,8), (8, npy.shape[0] -8), (npy.shape[1] -8,8),
                 (npy.shape[1] - 8, npy.shape[0] -8)]
    densities = [npy[pos[1], pos[0]] for pos in positions]

    for position, density in zip(positions, densities):
        sym_x, sym_y = position[1], position[0]
        sym_box = [sym_y - size + 1, sym_x - size + 1, sym_y + size, sym_x + size]
        iou = get_iou(sym_box, nod_box)  # 判断是否与nod_box交并

        # by ljw判断周围像素值和+-差异是否达到15以上
        region_indices = np.s_[sym_x - size + 1:sym_x + size, sym_y - size + 1:sym_y + size]  # 指定区域
        tmp=npy[region_indices]
        # if_clear = np.all(npy[region_indices] < (250 / 255))
        # if iou == 0 and if_clear:
        #     break
        if iou == 0:
            break

    # assert iou == 0, 'Fail to find property way to add the +/- symbol.'
    cube_mark(npy, (sym_x, sym_y), size=size, fill_value=255)
    if iou!=0:
        plt.figure(figsize=(3, 3))
        plt.imshow(npy, cmap='gray')
        plt.title(str(malign))

        plt.axhline(y_min, color='red')
        plt.axhline(y_max, color='red')
        plt.axvline(x_min, color='red')
        plt.axvline(x_max, color='red')
        plt.show()
        print("a")

    # assert if_clear, 'Fail to find clear corner.'


    np.save(os.path.join(data_dir,f"BIG-JIT{str(picsize)}-CUBE",f"{row['id']}.npy"), npy)



    # id=row['id']
    # dest_filepath = os.path.join(data_dir, 'UNRESIZE', f'{id}.npy')
    # img_mask_out_path = os.path.join(data_dir, 'UNRESIZE-MASK', f'{id}.npy')
    # jit_img_savepath = os.path.join(data_dir, 'BIG-JIT', f'{id}.npy')
    # jit_mask_savepath = os.path.join(data_dir, 'BIG-JIT-MASK', f'{id}.npy')
    # img = np.load(dest_filepath)
    # mask = np.load(img_mask_out_path)
    # jit_img = np.load(jit_img_savepath)
    # jit_mask = np.load(jit_mask_savepath)
    # # 创建画布和子图
    # fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    # # 第一张子图
    # axs[0].imshow(img)
    # # axs[0].imshow(mask, alpha=0.5)  # 绘制透明mask
    # axs[0].axvline(x=row['jit_xmin'], color='r', linestyle='--', linewidth=2)  # 绘制jit坐标线
    # axs[0].axvline(x=row['jit_xmax'], color='r', linestyle='--', linewidth=2)
    # axs[0].axhline(y=row['jit_ymin'], color='r', linestyle='--', linewidth=2)
    # axs[0].axhline(y=row['jit_ymax'], color='r', linestyle='--', linewidth=2)
    # xmin, ymin, xmax, ymax = cal_mask_boundaries(mask)
    # axs[0].axvline(x=xmin, color='g', linestyle='-', linewidth=2)  # 绘制jit坐标线
    # axs[0].axvline(x=xmax, color='g', linestyle='-', linewidth=2)
    # axs[0].axhline(y=ymin, color='g', linestyle='-', linewidth=2)
    # axs[0].axhline(y=ymax, color='g', linestyle='-', linewidth=2)
    #
    # axs[0].set_title('Original Image with Mask and Jit Box')
    #
    # # 第二张子图
    # axs[1].imshow(jit_img)
    # axs[1].imshow(jit_mask, alpha=0.5)  # 绘制透明jit_mask
    # # axs[1].axvline(x=row['jmask_xmin'], color='r', linestyle='--', linewidth=2)  # 绘制jit_mask坐标线
    # # axs[1].axvline(x=row['jmask_xmax'], color='r', linestyle='--', linewidth=2)
    # # axs[1].axhline(y=row['jmask_ymin'], color='r', linestyle='--', linewidth=2)
    # # axs[1].axhline(y=row['jmask_ymax'], color='r', linestyle='--', linewidth=2)
    # axs[1].set_title('Jit Image with Jit Mask')
    #
    #
    # axs[2].imshow(npy, cmap='gray')
    # axs[2].set_title(str(malign))
    #
    # axs[2].axhline(y_min, color='red')
    # axs[2].axhline(y_max, color='red')
    # axs[2].axvline(x_min, color='red')
    # axs[2].axvline(x_max, color='red')
    # plt.show()
    #
    # print("a")
#全部resize为96*96,然后保存到原来文件夹
def resize_bigjit(ori_csv_path,data_dir,size):
    df_resize = pd.read_csv(ori_csv_path)
    os.makedirs(os.path.join(data_dir, f'BIG-JIT{str(size)}'),exist_ok=True)
    os.makedirs(os.path.join(data_dir, f'BIG-JIT{str(size)}-MASK'), exist_ok=True)
    for index, row in df_resize.iterrows():
        id = row['id']
        jit_img_loadpath = os.path.join(data_dir, f'BIG-JIT', f'{id}.npy')
        jit_mask_loadpath = os.path.join(data_dir, f'BIG-JIT-MASK', f'{id}.npy')
        jit_img_savepath = os.path.join(data_dir,f'BIG-JIT{str(size)}', f'{id}.npy')
        jit_mask_savepath = os.path.join(data_dir, f'BIG-JIT{str(size)}-MASK', f'{id}.npy')

        img = cv2.resize(np.load(jit_img_loadpath),(size,size),interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(np.load(jit_mask_loadpath),(size,size),interpolation=cv2.INTER_NEAREST)

        df_resize.at[index, 'jmask_xmin'], df_resize.at[index, 'jmask_ymin'], df_resize.at[index, 'jmask_xmax'], \
        df_resize.at[index, 'jmask_ymax'] = cal_mask_boundaries(mask)
        np.save(jit_img_savepath, img)
        np.save(jit_mask_savepath, mask)

        # id=row['id']
        # dest_filepath = os.path.join(data_dir, 'UNRESIZE', f'{id}.npy')
        # img_mask_out_path = os.path.join(data_dir, 'UNRESIZE-MASK', f'{id}.npy')
        # # jit_img_savepath = os.path.join(data_dir, 'BIG-JIT', f'{id}.npy')
        # # jit_mask_savepath = os.path.join(data_dir, 'BIG-JIT-MASK', f'{id}.npy')
        # img = np.load(dest_filepath)
        # mask = np.load(img_mask_out_path)
        # jit_img = np.load(jit_img_savepath)
        # jit_mask = np.load(jit_mask_savepath)
        # # 创建画布和子图
        # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        # # 第一张子图
        # axs[0].imshow(img)
        # # axs[0].imshow(mask, alpha=0.5)  # 绘制透明mask
        # axs[0].axvline(x=row['jit_xmin'], color='r', linestyle='--', linewidth=2)  # 绘制jit坐标线
        # axs[0].axvline(x=row['jit_xmax'], color='r', linestyle='--', linewidth=2)
        # axs[0].axhline(y=row['jit_ymin'], color='r', linestyle='--', linewidth=2)
        # axs[0].axhline(y=row['jit_ymax'], color='r', linestyle='--', linewidth=2)
        # xmin, ymin, xmax, ymax = cal_mask_boundaries(mask)
        # axs[0].axvline(x=xmin, color='g', linestyle='-', linewidth=2)  # 绘制jit坐标线
        # axs[0].axvline(x=xmax, color='g', linestyle='-', linewidth=2)
        # axs[0].axhline(y=ymin, color='g', linestyle='-', linewidth=2)
        # axs[0].axhline(y=ymax, color='g', linestyle='-', linewidth=2)
        #
        # axs[0].set_title('Original Image with Mask and Jit Box')
        #
        # # 第二张子图
        # axs[1].imshow(jit_img)
        # axs[1].imshow(jit_mask, alpha=0.5)  # 绘制透明jit_mask
        # axs[1].axvline(x=df_resize.at[index, 'jmask_xmin'], color='r', linestyle='--', linewidth=2)  # 绘制jit_mask坐标线
        # axs[1].axvline(x=df_resize.at[index, 'jmask_xmax'], color='r', linestyle='--', linewidth=2)
        # axs[1].axhline(y=df_resize.at[index, 'jmask_ymin'], color='r', linestyle='--', linewidth=2)
        # axs[1].axhline(y=df_resize.at[index, 'jmask_ymax'], color='r', linestyle='--', linewidth=2)
        # axs[1].set_title('Jit Image with Jit Mask')
        # plt.show()
        # print("a")
    df_resize.to_csv(os.path.join(data_dir,f'annos_{str(size)}bigjit.csv'), index=False)


if __name__ == '__main__':
    picsize=112
    ori_csv_path = "/data/lijingwen/preprocess_DDSM/02_PROCESSED/Val/annos_bigjit.csv"
    csv_path = f"/data/lijingwen/preprocess_DDSM/02_PROCESSED/Val/annos_{str(picsize)}bigjit.csv"
    data_dir = "/data/lijingwen/preprocess_DDSM/02_PROCESSED/Val"
    # resize_bigjit(ori_csv_path, data_dir,size=picsize)
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        # add_cross(row, data_dir,picsize=picsize)
        add_cube(row, data_dir, picsize=picsize)

    # csv_path = "/data/lijingwen/preprocess_DDSM/02_PROCESSED/Test/annos_bigjit.csv"
    # df = pd.read_csv(csv_path)
    # data_dir = "/data/lijingwen/preprocess_DDSM/02_PROCESSED/Test"
    # for index, row in df.iterrows():
    #     add_cross(row, data_dir)
    #     add_cube(row, data_dir, picsize=picsize)

    ori_csv_path = "/data/lijingwen/preprocess_DDSM/02_PROCESSED/Train/annos_bigjit.csv"
    csv_path = f"/data/lijingwen/preprocess_DDSM/02_PROCESSED/Train/annos_{picsize}bigjit.csv"
    data_dir = "/data/lijingwen/preprocess_DDSM/02_PROCESSED/Train"
    df = pd.read_csv(csv_path)

    # resize_bigjit(ori_csv_path, data_dir,size=picsize)
    for index, row in df.iterrows():
        # add_cross(row, data_dir,picsize=picsize)
        add_cube(row, data_dir, picsize=picsize)


