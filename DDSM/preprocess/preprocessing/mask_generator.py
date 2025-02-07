import io
import os
import cv2
import plistlib

import numpy as np
import pydicom

from skimage.draw import polygon
from PIL import Image

from utils.config_ljw import LOGGING_DATA_PATH, CBIS_DDSM_DB_PATH
from utils.functions import get_path, get_filename, search_files, get_value_from_args_if_exists



def get_mias_roi_mask(args) -> None:
    """
    Función para obtener las máscaras del set de datos MIAS
    :param args: Lista de argumentos que contendra según las siguientes posiciones:
            1 - Path de la imagen original
            2 - coordenada X de la lesion (eje de coordenadas margen inferior izquierdo de la imagen)
            3 - Coordenada Y de la lesión (eje de coordenadas margen inferior izquierdo de la imagen)
            4 - Radio del circulo que contiene la lesión.
    """

    try:
        if len(args) != 4:
            raise ValueError('Incorrect number of args for function get_mias_roi')

        assert not os.path.isfile(args[0]), f'Mask {args[0]} already created.'

        # Se crea una máscara de tamaño 1024x1024 (resolución de las imágenes de MIAS).
        mask = np.zeros(shape=(1024, 1024), dtype=np.uint8)
        # Se dibuja un circulo blanco en las coordenadas indicadas. Si una imagen contiene multiples ROI's, se dibujan
        # todas las zonas.
        for x, y, rad in zip(args[1], args[2], args[3]):
            cv2.circle(mask, center=(int(x), int(y)), radius=int(rad), thickness=-1, color=(255, 255, 255))
        cv2.imwrite(args[0], mask)

    except AssertionError as err:
        print(f'{"=" * 100}\nAssertion Error in image processing\n{err}\n{"=" * 100}')

    except ValueError as err:
        print(f'{"=" * 100}\nError calling function get_inbrest_roi_mask pipeline\n{err}\n{"=" * 100}')

    except Exception as err:
        print(f'{"=" * 100}\n{get_filename(get_filename(args[0]))}\n{err}\n{"=" * 100}')


def get_cbis_roi_mask(args) -> None:
    """
    Función para obtener las máscaras del set de datos CBIS - DDSM
    :param args: Lista de argumentos que contendra según las siguientes posiciones:
            1 - Path de la imagen original
            2 - Path para guardar la máscara generada
    """
    try:
        if len(args) != 2:
            raise ValueError('Incorrect number of arguments for function get_cbis_roi_mask')

        assert not os.path.isfile(args[1]), f'Mask {args[1]} already created.'

        # Dado que una misma imagen puede contener multiples lesiones informadas mediante sufijos _N siendo N un entero
        # se recuperan todas las máscaras de ROIS para una misma imágen mamográfica.
        masks = []
        for img in search_files(file=get_path(CBIS_DDSM_DB_PATH, f'{args[0]}*_[0-9]'), ext='dcm'):

            # Se lee la información de las imagenes en formato dcm
            img = pydicom.dcmread(img)

            # Se convierte las imagenes a formato de array
            img_array = img.pixel_array.astype(float)

            # Las imagenes binarias únicamente pueden contener dos valroes de pixel distintos
            if len(np.unique(img_array)) == 2:

                # Se realiza un reescalado de la imagen para obtener los valores entre 0 y 255
                rescaled_image = (np.maximum(img_array, 0) / max(img_array)) * 255

                # Se limpian mascaras que sean de menos de 10 píxeles
                _, _, h, w = cv2.boundingRect(np.uint8(rescaled_image))

                if (h > 10) and (w > 10):
                    # Se convierte la imagen al ipode datos unsigned de 8 bytes
                    masks.append(np.uint8(rescaled_image))

        # Las distintas mascaras se sumarán para obtener una única máscara por mamografia
        final_mask = sum(masks)
        final_mask[final_mask > 1] = 255

        # Se almacena la mascara
        cv2.imwrite(args[1], final_mask)

    except AssertionError as err:
        print(f'{"=" * 100}\nAssertion Error in image processing\n{err}\n{"=" * 100}')

    except ValueError as err:
       print(f'{"=" * 100}\nError calling function get_cbis_roi_mask pipeline\n{err}\n{"=" * 100}')

    except Exception as err:
        print(f'{"=" * 100}\n{get_filename(args[1])}\n{err}\n{"=" * 100}')


def get_test_mask(args) -> None:
    """
    在应用程序的部署中绘制掩码的函数
    :param args: 参数列表，根据以下位置包含：
            1 - 原始图像路径
            2 - 输出掩码路径
            3 - 损伤的X坐标（图像左下角坐标轴）
            4 - 损伤的Y坐标（图像左下角坐标轴）
            5 - 包含损伤的圆的半径
            6 - 存储产生错误的目录的文件路径
    """

    error_path: io = get_value_from_args_if_exists(args, 5, LOGGING_DATA_PATH, IndexError, KeyError)

    try:
        if len(args) < 5:
            raise ValueError('get_mias_roi 函数的参数数量不正确')

        img_io_in = args[0]
        mask_io_out = args[1]
        x_cord = args[2]
        y_cord = args[3]
        rad = args[4]

        # 检查原始图像是否存在
        if not os.path.isfile(img_io_in):
            raise FileNotFoundError(f'{img_io_in} 未找到')

        # 检查是否先前创建了掩码
        assert not os.path.isfile(mask_io_out), f'掩码 {mask_io_out} 已经存在'

        # 原始图像的尺寸
        shape = cv2.imread(img_io_in).shape[:2]

        # 验证坐标 x、y 和半径
        if not 0 <= x_cord <= shape[1]:
            raise ValueError(f'{x_cord} 超出图像可用像素范围')

        if not 0 <= y_cord <= shape[0]:
            raise ValueError(f'{x_cord} 超出图像可用像素范围')

        if rad <= 0:
            raise ValueError(f'{rad} 的值不正确')

        # 根据cv2的坐标系修改Y坐标
        y_cord = shape[0] - y_cord

        # 创建具有原始图像分辨率的黑色图像
        mask = np.zeros(shape=shape, dtype=np.uint8)
        # 在每张乳房X兴趣区域绘制一个白色圆
        cv2.circle(mask, center=(int(x_cord), int(y_cord)), radius=int(rad), thickness=-1, color=(255, 255, 255))
        cv2.imwrite(mask_io_out, mask)

    except AssertionError as err:
        print(f'{"=" * 100}\n图像处理中的断言错误\n{err}\n{"=" * 100}')

    except ValueError as err:
        print(f'{"=" * 100}\n调用 get_test_mask pipeline 函数时出错\n{err}\n{"=" * 100}')

    except Exception as err:
        print(f'{"=" * 100}\n{get_filename(get_filename(args[0]))}\n{err}\n{"=" * 100}')

