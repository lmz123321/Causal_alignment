from glob import glob
from pathlib import Path
from typing import IO as io
from typing import Union, Any

import os
import numpy as np
import cv2
import math
import logging
import traceback
import pygogo
import pandas as pd

# 获取文件名
def get_filename(x: io) -> str:
    """
    获取文件名的函数
    :param x: 文件路径
    :return: 文件名
    """
    return os.path.basename(os.path.splitext(x)[0])

# 获取包含文件的目录名
def get_dirname(x: io) -> str:
    """
    获取包含文件的目录名的函数
    :param x: 文件路径
    :return: 目录名
    """
    return os.path.dirname(os.path.splitext(x)[0])

# 创建路径并可选地创建目录
def get_path(*args: Union[io, str], create: bool = True) -> io:
    """
    创建路径的函数，对不同操作系统进行了适配
    :param args: 构成路径的目录名
    :param create: 如果目录不存在是否创建
    :return: 最终的文件路径
    """
    path = os.path.join(*args)
    if create:
        create_dir(get_dirname(path))
    return path

# 创建目录及其所有父目录（如果不存在）
def create_dir(path: io):
    """
    创建目录及其所有父目录的函数
    :param path: 要创建的目录
    """
    Path(path).mkdir(parents=True, exist_ok=True)

# 在目录中查找具有指定扩展名的所有文件
def search_files(file: io, ext: str, in_subdirs: bool = True) -> iter:
    """
    在目录中查找具有指定扩展名的所有文件的函数
    :param file: 进行搜索的目录
    :param ext: 文件扩展名
    :param in_subdirs: 是否在子目录中搜索
    :return: 具有指定扩展名的文件列表
    """
    if in_subdirs:
        return glob(os.path.join(file, '**', f'*.{ext}'), recursive=True)
    else:
        return glob(os.path.join(file, f'*.{ext}'), recursive=True)

# 保存图像
def save_img(img: np.ndarray, save_example_dirpath: io, name: str):
    """
    保存图像的函数
    :param img: 要保存的图像
    :param save_example_dirpath: 保存图像的目录
    :param name: 图像的名称
    """
    if save_example_dirpath is not None:
        cv2.imwrite(get_path(save_example_dirpath, f'{name}.png'), img=img)

# 检测函数错误的装饰器
def detect_func_err(func):
    """
    检测函数错误的装饰器
    """
    def _exec(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as err:
            err.args += (func.__name__, )
            raise

    return _exec

# 找到最接近的2的幂次方
def closest_power2(x):
    """
    返回最接近的2的幂次方，通过检查第二个二进制数字是否为1。
    """
    op = math.floor if bin(x)[3] != "1" else math.ceil
    return 2**(op(math.log(x, 2)))

# 获取神经元数量
def get_number_of_neurons(previous_shape: list) -> int:
    """
    获取神经元层的最佳数量的函数，该数量是先前层神经元数量平方根的最接近整数值
    :param previous_shape: 先前层的神经元数量
    :return: 最佳神经元数量
    """
    num_params = np.prod(list(filter(lambda x: x is not None, previous_shape)))
    return closest_power2(int(np.sqrt(num_params)) - 1)

# 将数据批量写入文件
def bulk_data(file: io, mode: str = 'w', **kwargs) -> None:
    """
    将关键字参数写入文件的函数
    :param file: 要写入的文件名
    :param mode: 文件写入模式（W: 写入或A: 追加）
    :param kwargs: 键将成为文件的列，值将成为要写入的列的值
    """
    pd.DataFrame.from_dict(kwargs, orient='index').T.\
        to_csv(file, sep=';', decimal=',', header=not os.path.isfile(file) or mode == 'w', mode=mode, encoding='utf-8',
               index=False)

# 如果存在，从参数中获取值
def get_value_from_args_if_exists(args: list, pos: int, default: Any, *exceptions) -> Any:
    """
    从一组参数中获取参数值，如果出现指定的异常，则返回默认值
    :param args: 要搜索的参数列表
    :param pos: 参数位置
    :param default: 默认值
    :param exceptions: 要处理的异常
    :return: 位置对应的参数值
    """
    try:
        return args[pos]
    except tuple(exceptions):
        return default

# 创建文件格式化程序
def create_file_formatter(uri_log):
    """
    创建用于日志的csv文件，具有头部信息
    """
    # 获取不包括文件名的路径
    get_path(uri_log, create=True)

    # 如果日志不存在，创建头部信息
    if not os.path.isfile(uri_log):
        with open(uri_log, mode="w") as file:
            file.write("; ".join(['Fecha', 'Modulo', 'File', 'Función', 'Linea', 'Descripción', 'Tipo Error', 'Error']))
            file.write("\n")

    return logging.FileHandler(uri_log)

# 转换错误参数
def transform_error(**kwargs_in):
    """
    处理异常参数的函数。
    :param kwargs_in: 字典的键。
            'error': 异常消息
            'description': 用户编写的故事，以更好地理解程序的错误

    :return: 包含以下参数的字典:
            'func_name': 产生异常的函数名。
            'error_name': 错误类型的名称。
            'error': 错误描述。
            'error_line': 产生错误的代码行
            'description': 用户编写的故事，以更好地理解程序的错误。
    """

    kwargs_out = {
        "file_name": "",
        "func_name": "",
        "error_line": "",
        "error_name": "",
        "error": "",
        "description": ""
    }

    if "error" in kwargs_in.keys():
        stack_info = traceback.extract_tb(kwargs_in['error'].__traceback__)[-1]
        kwargs_out['file_name'] = f'{get_filename(stack_info[0])}'
        kwargs_out['func_name'] = stack_info[2]
        kwargs_out['error_line'] = stack_info[1]
        kwargs_out["error_name"] = kwargs_in["error"].__class__.__name__
        kwargs_out["error"] = str(kwargs_in["error"]).replace("\n", " | ").replace(";", ":")

    if "description" in kwargs_in.keys():
        kwargs_out["description"] = kwargs_in['description']

    return kwargs_out

# 记录错误的函数
def log_error(module, file_path, **kwargs_msg):
    """
    生成错误日志的函数。
    :param module: 生成错误的脚本和模块的名称
    :param file_path: 将生成错误文件的路径
    :param kwargs_msg:
            'error': 生成的异常
            'description': 用户编写的故事，以更好地理解程序的错误。
    """
    logging_fmt = '%(asctime)s;%(name)s;%(message)s'
    fmttr = logging.Formatter(logging_fmt, datefmt=pygogo.formatters.DATEFMT)
    fhdlr = create_file_formatter(file_path)

    logger = pygogo.Gogo(name=module, high_hdlr=fhdlr, high_formatter=fmttr, monolog=True).get_logger("py")

    kwargs_error = transform_error(**kwargs_msg)

    msg = "{file_name};{func_name};{error_line};{description};{error_name};{error}".format(**kwargs_error)

    if len(logger.handlers) > 2:
        logger.handlers.pop(0)
    logger.error(msg)

    for hdlr in logger.handlers:
        hdlr.close()
        logger.removeHandler(hdlr)

# 加载点
def load_point(point_string: str) -> tuple:
    """
    给定一个带有元组 (x, y) 的字符串，获取 x 和 y 的值
    :param point_string: 包含元组的字符串
    :return: 包含转换为浮点数的 y 和 x 坐标的元组
    """
    x, y = tuple([float(num) for num in point_string.strip('()').split(',')])
    return y, x

# 获取轮廓
def get_contours(img: np.ndarray) -> list:
    """
    获取给定掩码的轮廓的函数
    :param img: 图像
    :return: 图像的轮廓
    """
    return cv2.findContours(image=img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)[0]



def excel_column_name(n: int) -> str:
    """
    Función para convertir un número en su respectiva letra de columna para excel
    :param n: número de la columna
    :return: letra de la columna
    """
    name = ''
    while n > 0:
        n, r = divmod(n-1, 26)
        name = chr(r + ord('A')) + name
        return name