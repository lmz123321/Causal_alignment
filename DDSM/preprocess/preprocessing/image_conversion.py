import os
import sys

import pydicom
import numpy as np

from pathlib import Path

from PIL import Image

from utils.functions import get_path, get_filename, get_dirname, get_value_from_args_if_exists
from utils.config_ljw import CBIS_DDSM_DB_PATH




def convert_img(args):
    """
    负责将接收到的图像从接收到的格式转换为明确的格式的函数。

    :param args: 参数应包括：
        - 位置 0：（必需）要转换的图像路径。
        - 位置 1：（必需）转换后的图像路径。
    """
    exception_file=None
    try:

        if not (len(args) >= 2):
            raise ValueError('convert_dcm_img 函数的参数不足。最少需要 3 个必需参数')

        img_path = os.path.join(CBIS_DDSM_DB_PATH, args[0])
        dest_path = args[1]


        # 验证转换格式是否正确，并验证要转换的图像是否存在
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"{img_path} 不存在。")
        if os.path.splitext(img_path)[1] not in ['.pgm', '.dcm']:
            raise ValueError('只能转换为以下格式：pgm，dcm')

        assert not os.path.isfile(dest_path), f'已存在转换后的图像 {dest_path}'

        # 根据原始格式执行不同的转换
        if os.path.splitext(img_path)[1] == '.dcm':
            exception_file=convert_dcm_imgs(ori_path=img_path, dest_path=dest_path)
        else:
            raise KeyError(f'{os.path.splitext(img_path)} 的转换函数尚未实现')

    except AssertionError as err:
        print(f'{"=" * 100}\nconvert_img 中的断言错误\n{err}\n{"=" * 100}')

    except Exception as err:
        print(f'{"=" * 100}\n{get_filename(img_path)}\n{err}\n{"=" * 100}')
    return exception_file


def convert_dcm_imgs(ori_path, dest_path):
    """
    读取 dcm 格式图像并将其转换为用户指定格式的函数。

    :param ori_path: 图像的原始路径
    :param dest_path: 图像的目标路径

    """
    try:
        if os.path.splitext(dest_path)[1] not in ['.png', '.jpg','.npy']:
            raise ValueError('only transform into：png，jpg，npy')
        Path(get_dirname(dest_path)).mkdir(parents=True, exist_ok=True)
        img = pydicom.dcmread(ori_path)
        img_array = img.pixel_array.astype(float)
        rescaled_image = (np.maximum(img_array, 0) / np.max(img_array)) * 255
        final_image = np.uint8(rescaled_image)
        np.save(dest_path,final_image)
        # print(dest_path)
        return None


    except AssertionError:
        return ori_path


