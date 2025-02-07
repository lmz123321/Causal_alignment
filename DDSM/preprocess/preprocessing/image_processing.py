import numpy as np
import cv2
import os
import sys

from itertools import repeat
from typing import IO as io

import pydicom
from PIL import Image
from matplotlib import pyplot as plt

from preprocessing.functions import (
    apply_clahe_transform, remove_artifacts, remove_noise, crop_borders, pad_image_into_square, resize_img,
    normalize_breast, flip_breast, correct_axis, remove_artifacts_crop
)
from utils.config_ljw import LOGGING_DATA_PATH, PREPROCESSING_FUNCS, PREPROCESSING_CONFIG
from utils.functions import (
    get_filename,get_contours
)



def full_image_pipeline(img_filepath,dest_filepath,img_mask_filepath,img_mask_out_path,if_resize=True):
    """
    用于执行乳腺X射线摄影图像预处理的函数。该预处理包括：
        1 - 裁剪图像边缘。
        2 - 消除噪音。
        3 - 去除在图像上做的标记。
        4 - 进行最小-最大归一化，将图像标准化为8位。
        5 - 对图像进行直方图均衡化以提高对比度。
        6 - 进行水平翻转以标准化乳房的方向。
        7 - 对图像进行填充，使其具有所需的纵横比。
        8 - 调整图像大小。
    如果存在掩码，则将应用以下区域感兴趣的功能（用于数据分割的功能）：
        1 - 裁剪图像边缘。
        2 - 进行水平翻转以标准化乳房的方向。
        3 - 对图像进行填充，使其具有所需的纵横比。
        4 - 调整图像大小。

    :param args: 参数列表，位置应为：
        1 - 未经处理的图像路径。
        2 - 处理后的图像保存路径。
        3 - 用于在图形上表示中间步骤的布尔值。
        4 - 处理后的包含感兴趣区域掩码的路径。
        5 - 每个图像感兴趣区域的掩码的原始路径。
    """

    try:


        if_exception = False
        # 验证转换格式是否正确，并验证原始图像是否存在。
        if not os.path.isfile(img_filepath):
            raise FileNotFoundError(f'图像{img_filepath}不存在。')
        if os.path.splitext(dest_filepath)[1] not in ['.png', '.jpg','.npy']:
            raise ValueError(f'仅支持png、jpg、npy的转换')

        if os.path.isfile(dest_filepath):
            print(f'已存在处理文件：{dest_filepath}')
            return if_exception

        # 存储预处理配置
        prep_dict = PREPROCESSING_FUNCS[PREPROCESSING_CONFIG]

        # 读取未处理的原始图像。
        img = np.load(img_filepath)

        # 读取掩码
        if img_mask_filepath is None:
            img_mask = np.ones(shape=img.shape, dtype=np.uint8)
        else:
            if not os.path.isfile(img_mask_filepath):
                raise FileNotFoundError(f'掩码{img_mask_filepath}不存在。')
            dicom_data = pydicom.dcmread(img_mask_filepath)
            pixel_array = dicom_data.pixel_array
            # 将像素数组二值化
            _, img_mask = cv2.threshold(pixel_array, 0, 255, cv2.THRESH_BINARY)


        images = {'ORIGINAL': img}

        # 首先，对完整图像进行裁剪
        images['CROPPING 1'] = crop_borders(images[list(images.keys())[-1]].copy(), **prep_dict.get('CROPPING_1', {}))

        # 将相同的处理应用于掩码
        img_mask = crop_borders(img_mask, **prep_dict.get('CROPPING_1', {}))

        # 然后，通过使用中值滤波器消除图像的噪音
        images['REMOVE NOISE'] = remove_noise(
            img=images[list(images.keys())[-1]].copy(), **prep_dict.get('REMOVE_NOISE', {}))

        # 接下来是去除图像的伪影。
        images['REMOVE ARTIFACTS'], _, mask, img_mask = remove_artifacts(
            img=images[list(images.keys())[-1]].copy(), mask=img_mask, **prep_dict.get('REMOVE_ARTIFACTS', {})
        )

        # plt.imshow(images['REMOVE ARTIFACTS'], cmap='gray')
        # plt.imshow(mask, cmap='gray', alpha=0.5)
        # save_dir = os.path.join('/'.join(img_mask_out_path.split('/')[:-2]), 'chest_mask')
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        # save_seg = os.path.join(save_dir,img_mask_out_path.split('/')[-1])
        # plt.savefig(f'{save_seg}.png')
        # # # plt.show()


        # 伪影去除后，对胸部区域进行归一化处理
        images['IMAGE NORMALIZED'] = \
            normalize_breast(images[list(images.keys())[-1]].copy(), mask, **prep_dict.get('NORMALIZE_BREAST', {}))

        # 然后，应用一系列图像均衡化。最多可以应用3次均衡化，每次均以各自的通道表示
        ecual_imgs = []
        img_to_ecualize = images[list(images.keys())[-1]].copy()
        assert 0 < len(prep_dict['ECUALIZATION'].keys()) < 4, '不正确的均衡化次数'
        for i, (ecual_func, ecual_kwargs) in enumerate(prep_dict['ECUALIZATION'].items(), 1):

            if 'CLAHE' in ecual_func.upper():
                images[ecual_func.upper()] = apply_clahe_transform(img=img_to_ecualize, mask=mask, **ecual_kwargs)
                ecual_imgs.append(images[list(images.keys())[-1]].copy())

            elif 'GCN' in ecual_func.upper():
                pass

        if len(prep_dict['ECUALIZATION'].keys()) == 2:
            images['IMAGES SYNTHESIZED'] = cv2.merge((img_to_ecualize, *ecual_imgs))
        elif len(prep_dict['ECUALIZATION'].keys()) == 3:
            images['IMAGES SYNTHESIZED'] = cv2.merge(tuple(ecual_imgs))

        # 对图像进行水平翻转（如果有必要）：
        images['IMG_FLIP'], flip = flip_breast(images[list(images.keys())[-1]].copy(), **prep_dict.get('FLIP_IMG', {}))

        if flip:
            img_mask = cv2.flip(src=img_mask, flipCode=1)

        # 最后进行左侧的裁剪
        # images['CROPPING LEFT'] = crop_borders(images[list(images.keys())[-1]].copy(),
        #                                        **prep_dict.get('CROPPING_2', {}))
        # img_mask = crop_borders(img=img_mask,  **prep_dict.get('CROPPING_2', {}))

        # # 对图像进行填充，以将其转换为具有所需纵横比的图像
        # if prep_dict.get('RATIO_PAD', False):
        #     images['IMAGE RATIO PADDED'] = \
        #         pad_image_into_square(img=images[list(images.keys())[-1]].copy(), **prep_dict.get('RATIO_PAD', {}))
        #     img_mask = pad_image_into_square(img=img_mask, **prep_dict.get('RATIO_PAD', {}))


        if if_resize is True and prep_dict.get('RESIZING', False):
            A=images[list(images.keys())[-1]].copy()
            resize_shape=prep_dict.get('RESIZING', {})
            a, b = A.shape
            if a >= b:
                new_a = resize_shape['height']
                new_b = int(b * (new_a / a))
            else:
                new_b = resize_shape['height']
                new_a = int(a * (new_b / b))

            images['IMAGE RESIZED'] =cv2.resize(A, (new_b, new_a),interpolation=cv2.INTER_NEAREST)
            img_mask = cv2.resize(img_mask, (new_b, new_a),interpolation=cv2.INTER_NEAREST)
            # resize_img(img=A, **prep_dict.get('RESIZING', {}), interpolation=cv2.INTER_NEAREST)
            # img_mask = resize_img(img=img_mask, **prep_dict.get('RESIZING', {}), interpolation=cv2.INTER_NEAREST)
            # images['IMAGE RESIZED'] = \
            #     resize_img(img=images[list(images.keys())[-1]].copy(), **prep_dict.get('RESIZING', {}))
            # img_mask = resize_img(img=img_mask, **prep_dict.get('RESIZING', {}), interpolation=cv2.INTER_NEAREST)

        if img_mask_out_path and len(get_contours(img_mask)) > 0:
            np.save(img_mask_out_path,np.uint8(img_mask))

        # 存储最终图像
        np.save(dest_filepath,np.uint8(images[list(images.keys())[-1]].copy()))

        return if_exception



    except IndexError as err:
        if_exception = True
        print(f'{"=" * 100}\n调用convert_dcm_img pipeline函数时的错误\n{err}\n{"=" * 100}')
        return if_exception

    except Exception as err:
        if_exception = True
        print(f'{"=" * 100}\n{get_filename(img_filepath)}\n{err}\n{"=" * 100}')
        return if_exception



def crop_image_pipeline(img_filepath,out_filepath,roi_mask_path):
    """
    用于对乳腺X光片中感兴趣区域进行预处理的函数。这个预处理包括：
        1 - 裁剪完整图像的边缘，以便消除感兴趣区域的边缘
        2 - 去除噪音
        3 - 去除在完整图像上做的标注以获得乳腺区域
        4 - 根据完整乳腺的掩模进行每个感兴趣区域的裁剪
        5 - 应用CLAHE均衡化以改善对比度
    :param args: 参数列表，其位置应为：
        1 - 未处理图像的路径
        2 - 处理后图像的目标路径
        3 - 兴趣区域掩模的来源路径
        -- 可选参数 --
        4 - 背景图像数量（无损伤区域）
        5 - 感兴趣区域图像数量
        6 - 感兴趣区域的重叠度
        7 - 每个感兴趣区域的额外裁剪边距
        8 - 布尔值，用于恢复中间步骤
    """

    try:
    # if 1:
        if_exception = False

        # n_background_imgs: int = get_value_from_args_if_exists(args, 3, 0, IndexError, TypeError)
        # n_roi_imgs: int = get_value_from_args_if_exists(args, 4, 1, IndexError, TypeError)
        # overlap_roi: float = get_value_from_args_if_exists(args, 5, 1.0, IndexError, TypeError)
        margin_roi=1.2
        # save_intermediate_steps: bool = get_value_from_args_if_exists(args, 7, False, IndexError, TypeError)

        # 确认转换格式是否正确，并验证要转换的图像是否存在
        if not os.path.isfile(img_filepath):
            print(f'图像{img_filepath}不存在。')
        if not os.path.isfile(roi_mask_path):
            print(f'图像{roi_mask_path}不存在。')
        if os.path.exists(out_filepath):
            print(f'图像{out_filepath}已存在。')
            return if_exception

        # 存储预处理配置
        prep_dict = PREPROCESSING_FUNCS[PREPROCESSING_CONFIG]

        # 读取原始未处理图像。
        img = np.load(img_filepath)
        mask = pydicom.dcmread(roi_mask_path).pixel_array


        # 首先，对图像进行裁剪，如果它们是完整图像的话
        crop_img = crop_borders(img, **prep_dict.get('CROPPING_1', {}))

        # 对掩模应用相同的处理
        img_mask = crop_borders(mask, **prep_dict.get('CROPPING_1', {}))

        # 然后，使用中值滤波器去除图像噪声
        img_denoised = remove_noise(crop_img, **prep_dict.get('REMOVE_NOISE', {}))

        # 从掩模中获取病变区域以及背景区域
        _, _, breast_mask, mask = remove_artifacts_crop(img_denoised, img_mask, False,
                                                   **prep_dict.get('REMOVE_ARTIFACTS', {}))

        # 获取病变区域的图块。
        roi_zones = []
        mask_zones = []
        breast_zone = breast_mask.copy()
        for contour in get_contours(img=mask):
            x, y, w, h = cv2.boundingRect(contour)

            if (h > 15) & (w > 15):
                center = (y + h // 2, x + w // 2)
                y_min, x_min = int(center[0] - h * margin_roi // 2), int(center[1] - w * margin_roi // 2)
                y_max, x_max = int(center[0] + h * margin_roi // 2), int(center[1] + w * margin_roi // 2)
                x_max, x_min, y_max, y_min = correct_axis(img_denoised.shape, x_max, x_min, y_max, y_min)
                roi_zones.append(img_denoised[y_min:y_max, x_min:x_max])
                mask_zones.append(breast_zone[y_min:y_max, x_min:x_max])

                # 删除病变区域以便后续获取背景区域
                cv2.rectangle(breast_mask, (x_min, y_min), (x_max, y_max), color=(0, 0, 0), thickness=-1)

        # 处理裁剪的感兴趣区域
        for idx, (roi, roi_mask, tipo) in enumerate(zip(roi_zones, mask_zones, repeat('roi', len(roi_zones)))):

            roi_norm = normalize_breast(roi, roi_mask, **prep_dict.get('NORMALIZE_BREAST', {}))

            # 接下来，对图像应用一系列均衡化处理。最多可应用3次均衡化，并分别应用于每个通道
            ecual_imgs = []
            img_to_ecualize = roi_norm.copy()
            assert 0 < len(prep_dict['ECUALIZATION'].keys()) < 4, 'Incorrect number of equalizations'
            for i, (ecual_func, ecual_kwargs) in enumerate(prep_dict['ECUALIZATION'].items(), 1):


                if 'CLAHE' in ecual_func.upper():
                    ecual_imgs.append(apply_clahe_transform(img_to_ecualize, roi_mask, **ecual_kwargs))

                elif 'GCN' in ecual_func.upper():
                    pass

            if len(prep_dict['ECUALIZATION'].keys()) == 2:
                roi_synthetized = cv2.merge((img_to_ecualize, *ecual_imgs))
            elif len(prep_dict['ECUALIZATION'].keys()) == 3:
                roi_synthetized = cv2.merge(tuple(ecual_imgs))
            else:
                roi_synthetized = ecual_imgs[-1]

            # 存储最终图像
            np.save(out_filepath, np.uint8(roi_synthetized).copy())

            return if_exception

    except IndexError as err:
        if_exception = True
        print(f'{"=" * 100}\n生成crop_img pipeline函数时的错误\n{err}\n{"=" * 100}')
        return if_exception

    except Exception as err:
        if_exception = True
        print(f'{"=" * 100}\n{get_filename(img_filepath)}\n{err}\n{"=" * 100}')
        return if_exception





