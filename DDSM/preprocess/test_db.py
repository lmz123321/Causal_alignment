import concurrent
import json
import sys
sys.path.append('/home/lijingwen/Projects/Counter_align_old/baseline/breast_cancer_diagnosis-master/src')
from tqdm import tqdm
from preprocessing.image_processing import full_image_pipeline, crop_image_pipeline
import os
import cv2

import numpy as np
import pandas as pd
import pydicom

from PIL import Image
from preprocessing.image_conversion import convert_img
from utils.functions import get_path, search_files, get_filename
from utils.config_ljw import TEST_CONVERTED_DATA_PATH, TEST_PREPROCESSED_DATA_PATH,\
    TRAIN_CONVERTED_DATA_PATH, TRAIN_PREPROCESSED_DATA_PATH,\
    VAL_CONVERTED_DATA_PATH, VAL_PREPROCESSED_DATA_PATH,\
    CBIS_DDSM_DB_PATH

class Datasets:

    """
        用于根据用户在应用程序中输入的Excel生成数据集的类
    """

    __name__ = 'SETS'
    # CSV_COLS = ['patient_id','breast_density','left or right breast','image view','abnormality id',
    # 'mass shape','mass margins','assessment','pathology','subtlety',
    # 'image file path','cropped image file path','ROI mask file path']
    CSV_COLS = ['patient_id', 'left or right breast', 'image view', 'abnormality id',
                'assessment', 'pathology', 'subtlety',
                'image file path', 'cropped image file path', 'ROI mask file path']

    def __init__(self, xlsx_io, out_path,mode):
        """
        类的初始化。
        :param xlsx_io: 将被读取的Excel的路径。
        :param signal: 用于记录可能的错误的信号。
        :param out_path: 将存储生成的输出的目录。
        """
        self.out_path = out_path
        self.df = self.get_df_from_info_files(path=xlsx_io)
        self.mode=mode
        if mode=="train":
            self.CONVERSION_PATH = TRAIN_CONVERTED_DATA_PATH
            self.PROCESED_PATH = TRAIN_PREPROCESSED_DATA_PATH
        elif mode=="test":
            self.CONVERSION_PATH = TEST_CONVERTED_DATA_PATH
            self.PROCESED_PATH = TEST_PREPROCESSED_DATA_PATH
        elif mode=="val":
            self.CONVERSION_PATH = VAL_CONVERTED_DATA_PATH
            self.PROCESED_PATH = VAL_PREPROCESSED_DATA_PATH

    def get_df_from_info_files(self, path):
        """
        用于读取用户输入的Excel并处理它的函数。
        :param path: 要处理的Excel的路径。
        :return: 处理后的信息的DataFrame
        """
        # CSV_COLS = ['patient_id', 'breast_density', 'left or right breast', 'image view', 'abnormality id',
        #             'mass shape', 'mass margins', 'assessment', 'pathology', 'subtlety',
        #             'image file path', 'cropped image file path', 'ROI mask file path']

        CSV_COLS = ['patient_id', 'left or right breast', 'image view', 'abnormality id',
                    'assessment', 'pathology', 'subtlety',
                    'image file path', 'cropped image file path', 'ROI mask file path']


        # 读取包含数据集信息的Excel
        df = pd.read_csv(path, dtype=str).reset_index(drop=True)
        # 添加新列 'id' Mass-Test_P_00016_LEFT_CC_1   1是abnoamlity id
        df['id'] = df['cropped image file path'].apply(lambda x: x.split('/')[0])

        # 检查列名是否正确
        if not all([c in df.columns for c in self.CSV_COLS]):
            raise ValueError(f'不正确的列名。请检查Excel包含下列列值：{", ".join(self.CSV_COLS)}')

        # 验证通过后，将原始Excel的列分配给实例的列，以免删除它们。
        self.CSV_COLS = df.columns.values.tolist()

        # 检查id是否唯一
        if any(df['id'].value_counts() > 1):
            a=df['id']
            raise ValueError(f'在Excel中发现重复的id值')
        # duplicates = df[df.duplicated('id', keep=False)]
        #
        # if not duplicates.empty:
        #     # 如果 'id' 列有重复值
        #     grouped_data = duplicates.groupby('id')
        #
        #     for group_id, group_data in grouped_data:
        #         print(f"Data for ID: {group_id}")
        #         print(group_data)
        #         print("\n")
        # else:
        #     print("All 'id' values are unique.")


        # 删除null数据
        incorrect = df[df[CSV_COLS].isna().any(axis=1)]
        if len(incorrect) > 0:
            print(
                f'找到不正确的数据！\n删除了 {len(incorrect)} 个在 存在属性为空的不正确的值。' +
                '被删除的文件有：\n\t- 文件:' + "\n\t- 文件:".join(incorrect['id'].values.tolist())
            )
        df.drop(index=incorrect.index, inplace=True)

        # 删除路径无效的数据
        # 定义检查文件路径存在的函数
        def check_file_path_exists(file_path):
            return os.path.exists(os.path.join(CBIS_DDSM_DB_PATH,file_path))

        # 对每一列进行检查，保留存在的文件路径行
        mask = (df['image file path'].apply(check_file_path_exists) &
                df['ROI mask file path'].apply(check_file_path_exists))

        incorrect = df[~mask]
        if len(incorrect) > 0:
            print(
                f'找到不正确的数据！\n{len(incorrect)} 个image和mask file path不成对存在。' +
                '被删除的文件有：\n\t- 文件:' + "\n\t- 文件:".join(incorrect['id'].values.tolist())
            )
        df.drop(index=incorrect.index, inplace=True)

        return df



    def convert_images_format(self):
        """
        用于将图像从pgm或dicom格式转换为png格式的函数。
        :param signal: 用于表示处理进度的Pyqt信号
        :param min_value: 进度条的最小值。
        :param max_value: 进度条的最大值。
        """

        # 创建 CONVERTED_IMG 列，其中将转换格式的图像保存
        # /data/lijingwen/preprocess_DDSM/01_CONVERTED/Test/FULL/Mass-Test_P_00016_LEFT_CC_1.png

        #原图像npy格式生成
        self.df.loc[:, 'CONVERTED_IMG'] = self.df.apply(lambda x: get_path(self.CONVERSION_PATH, 'FULL-INITIAL', f'{x.id}.npy'),#todo ljw FULL->FULL-INITIAL
                                                        axis=1)
        exception_files = []


        for i, arg in enumerate(list(set(
                [(row['image file path'], row['CONVERTED_IMG'], False, self.out_path) for _, row in self.df.iterrows()])), 1):
            if i%20==0:
                print(f'转换图像img: {i}/{len(self.df)}/{self.mode}')
            e_file=convert_img(arg)
            if e_file is not None:
                exception_files.append(e_file)

        print(f'异常文件共：{len(exception_files)}个')
        mask = self.df['CONVERTED_IMG'].isin(exception_files)
        self.df = self.df[~mask]


        # 过滤异常文件
        mask = self.df['CONVERTED_IMG'].isin(exception_files)
        self.df = self.df[~mask]


        # # CROP jpg格式生成
        # self.df.loc[:, 'CROPPED_IMG'] = self.df.apply(lambda x: get_path(self.CONVERSION_PATH, 'CROP', f'{x.id}.npy'),axis=1)
        # exception_files = []
        # for i, arg in enumerate(list(set(
        #             [(row['cropped image file path'], row['CROPPED_IMG'], False, self.out_path) for _, row in self.df.iterrows()])), 1):
        #     if i % 20 == 0:
        #         print(f'转换图像crop {i}/{len(self.df)}/{self.mode}')
        #     e_file=convert_img(arg)
        #     if e_file is not None:
        #         exception_files.append(e_file)
        # print(f'异常文件共：{len(exception_files)}个')
        # mask = self.df['CROPPED_IMG'].isin(exception_files)
        # self.df = self.df[~mask]
        self.df=self.df.reset_index(drop=True)

    def calculate_mask_boundaries(self,mask_path):
        # 读取 mask 图像
        mask = np.load(mask_path)

        # 找到 mask 图像的上下左右边界
        non_zero_rows = cv2.findNonZero(mask)
        if non_zero_rows is not None:
            x_min, y_min = non_zero_rows.min(axis=0)[0]
            x_max, y_max = non_zero_rows.max(axis=0)[0]

        else:
            # 如果 mask 图像全为零，设置默认值或根据需求处理
            x_min, y_min = 0,0
            x_max, y_max = 0,0

        return x_min, y_min,x_max, y_max


    def calculate_maskdcm_boundaries(self,img_mask_filepath):
        # 读取 mask 图像
        dicom_data = pydicom.dcmread(img_mask_filepath)
        mask = dicom_data.pixel_array

        # 找到 mask 图像的上下左右边界
        non_zero_rows = cv2.findNonZero(mask)
        if non_zero_rows is not None:
            x_min, y_min = non_zero_rows.min(axis=0)[0]
            x_max, y_max = non_zero_rows.max(axis=0)[0]

        else:
            # 如果 mask 图像全为零，设置默认值或根据需求处理
            x_min, y_min = 0,0
            x_max, y_max = 0,0

        return x_min, y_min,x_max, y_max
    def save_coords(self):
        self.df=self.df.reset_index(drop=True)
        self.df['x_min'], self.df['y_min'], \
        self.df['x_max'], self.df['y_max'] = 0,0,0,0
        # 遍历 DataFrame 中的每一行
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Augmenting Images"):
            img_name = f"{row['id']}.npy"
            img_full_path = os.path.join(self.PROCESED_PATH,'FULL-INITIAL', img_name)#todo ljw FULL->FULL-INITIAL
            img_mask_path = os.path.join(self.PROCESED_PATH, 'MASK-INITIAL', img_name)

            #mask和img进行成对处理
            if os.path.exists(img_full_path) and os.path.exists(img_mask_path):
                a=self.calculate_mask_boundaries(mask_path=img_mask_path)
                self.df.loc[idx, 'x_min'] = a[0]
                self.df.loc[idx, 'y_min'] = a[1]
                self.df.loc[idx, 'x_max'] = a[2]
                self.df.loc[idx, 'y_max'] = a[3]

        return

    def map_atts(self):
        shape_dict = {
            'OVAL': 0,
            'OVAL-LYMPH_NODE': 0,
            'ROUND-OVAL': 0,
            'ROUND': 0,
            'LYMPH_NODE': 0,

            'IRREGULAR-ARCHITECTURAL_DISTORTION': 1,
            'ARCHITECTURAL_DISTORTION': 1,
            'IRREGULAR': 1,
            'LOBULATED-LYMPH_NODE': 1,
            'LOBULATED': 1,
            'LOBULATED-ARCHITECTURAL_DISTORTION': 1,
            'LOBULATED-IRREGULAR': 1,
            'LOBULATED-OVAL': 1,
            'ROUND-IRREGULAR-ARCHITECTURAL_DISTORTION': 1,
            'ROUND-LOBULATED': 1,

            'ASYMMETRIC_BREAST_TISSUE': None,  # 只可看见单侧，数据剔除
            'FOCAL_ASYMMETRIC_DENSITY': None,
            'IRREGULAR-FOCAL_ASYMMETRIC_DENSITY': None
        }
        sub_dict={'0':0,'1':0,'2':0,'3':1,'4':1,'5':1}

        self.df['mass shape'] = self.df['mass shape'].map(shape_dict)
        self.df['subtlety'] = self.df['subtlety'].map(sub_dict)
        self.df['CIRCUMSCRIBED'] = self.df['mass margins'].apply(lambda x: 1 if 'CIRCUMSCRIBED' in str(x) else 0)
        self.df['OBSCURED'] = self.df['mass margins'].apply(lambda x: 1 if 'OBSCURED' in str(x) else 0)
        self.df['ILL_DEFINED'] = self.df['mass margins'].apply(lambda x: 1 if 'ILL_DEFINED' in str(x) else 0)
        # self.df['MICROLOBULATED'] =self.df['mass margins'].apply(lambda x: 1 if 'MICROLOBULATED' in str(x) else 0)
        self.df['SPICULATED'] =self.df['mass margins'].apply(lambda x: 1 if 'SPICULATED' in str(x) else 0)

        self.df = self.df.dropna()




    def preprocess_image(self):
        print("PROCESSED img and mask...")

        # 创建一个空列表用于存储异常文件
        exception_ids = []
        exception_crop_ids = []

        for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Preprcessing Images"):
            tmp=row['id']
            get_path(self.PROCESED_PATH, 'FULL-INITIAL', f'{tmp}.npy')#todo ljw FULL->FULL-INITIAL
            img_filepath = os.path.join(self.CONVERSION_PATH, 'FULL-INITIAL', f'{tmp}.npy')#todo ljw FULL->FULL-INITIAL
            dest_filepath = os.path.join(self.PROCESED_PATH, 'FULL-INITIAL', f'{tmp}.npy')#todo ljw FULL->FULL-INITIAL

            get_path(self.PROCESED_PATH, 'MASK-INITIAL', f'{tmp}.npy')#todo ljw
            img_mask_filepath = os.path.join(CBIS_DDSM_DB_PATH,
                                             self.df[self.df['id'] == tmp]['ROI mask file path'].iloc[0])
            img_mask_out_path = os.path.join(self.PROCESED_PATH, 'MASK-INITIAL', f'{tmp}.npy')#todo ljw

            # # 图像处理
            # if_exception = full_image_pipeline(img_filepath, dest_filepath, img_mask_filepath, img_mask_out_path,if_resize=False)#by 241001 ljw if_resize=false
            #
            # if if_exception is True:
            #     exception_ids.append(tmp)
            #
            # get_path(self.PROCESED_PATH, 'CROP-INITIAL', f'{tmp}.npy')#todo lw

            if os.path.exists(img_filepath) is False or os.path.exists(dest_filepath) is False \
                or os.path.exists(img_mask_filepath) is False or os.path.exists(img_mask_out_path) is False:
                exception_ids.append(tmp)
            # #by ljw 241001 do not crop
            # # img_crop_filepath = os.path.join(self.CONVERSION_PATH, 'CROP-INITIAL',f'{tmp}.npy')
            # dest_crop_filepath = os.path.join(self.PROCESED_PATH, 'CROP-INITIAL', f'{tmp}.npy')#todo lw
            # if_crop_exception =crop_image_pipeline(img_filepath, dest_crop_filepath, img_mask_filepath)
            #
            # # if_crop_exception = full_image_pipeline(img_crop_filepath, dest_crop_filepath,None,None)
            # if if_crop_exception is True:
            #     exception_crop_ids.append(tmp)

        #记录边界框
        self.save_coords()


        # 删除对应有问题数据
        mask = self.df['id'].isin(exception_ids)
        self.df = self.df[~mask]
        print(f"FULL-INITIAL MASK-INITIAL preprocess异常共{len(exception_ids)}个：{exception_ids}")#todo lw
        mask = self.df['id'].isin(exception_crop_ids)
        self.df = self.df[~mask]
        print(f"CROP-INITIAL preprocess异常共{len(exception_crop_ids)}个：{exception_crop_ids}")#todo lw


def if_pathology(df):
    different_pathology_groups = df.groupby('image file path')['pathology'].nunique().reset_index()
    different_pathology_groups = different_pathology_groups[different_pathology_groups['pathology'] > 1]
    num_groups = len(different_pathology_groups)
    return num_groups

def map_atts(csv_path):
    df = pd.read_csv(csv_path)
    num_groups=if_pathology(df)
    shape_dict = {
        'OVAL': 0,
        'OVAL-LYMPH_NODE': 0,
        'ROUND-OVAL': 0,
        'ROUND': 0,
        'LYMPH_NODE': 0,

        'IRREGULAR-ARCHITECTURAL_DISTORTION': 1,
        'ARCHITECTURAL_DISTORTION': 1,
        'IRREGULAR': 1,
        'LOBULATED-LYMPH_NODE': 1,
        'LOBULATED': 1,
        'LOBULATED-ARCHITECTURAL_DISTORTION': 1,
        'LOBULATED-IRREGULAR': 1,
        'LOBULATED-OVAL': 1,
        'ROUND-IRREGULAR-ARCHITECTURAL_DISTORTION': 1,
        'ROUND-LOBULATED': 1,

        'ASYMMETRIC_BREAST_TISSUE': None,  # 只可看见单侧，数据剔除
        'FOCAL_ASYMMETRIC_DENSITY': None,
        'IRREGULAR-FOCAL_ASYMMETRIC_DENSITY': None
    }
    sub_dict = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    df['mass shape'] = df['mass shape'].map(shape_dict)
    df['subtlety'] = df['subtlety'].map(sub_dict)

    df['CIRCUMSCRIBED'] = df['mass margins'].apply(lambda x: 1 if 'CIRCUMSCRIBED' in str(x) else 0)
    df['OBSCURED'] = df['mass margins'].apply(lambda x: 1 if 'OBSCURED' in str(x) else 0)
    df['ILL_DEFINED'] = df['mass margins'].apply(lambda x: 1 if 'ILL_DEFINED' in str(x) else 0)
    # df['MICROLOBULATED'] = df['mass margins'].apply(lambda x: 1 if 'MICROLOBULATED' in str(x) else 0)
    df['SPICULATED'] = df['mass margins'].apply(lambda x: 1 if 'SPICULATED' in str(x) else 0)

    df = df.dropna()
    df.to_csv(csv_path)

def gen_json(csv_path,json_path):
    df=pd.read_csv(csv_path)
    data_dict = {'data': df[['id', 'pathology', 'subtlety','x_min', 'y_min', 'x_max','y_max','nids','pids'
                             ]].to_dict('records')}
    with open(json_path, 'w') as json_file:
        json.dump(data_dict, json_file)


def concat_df(df_path1,df_path2,out_path):
    # column_names=common_attributes = [
    #     "patient_id","breast_density","left or right breast","image view","abnormality id","abnormality type",
    #     "assessment","pathology","subtlety","image file path","cropped image file path","ROI mask file path",
    #     "id", "x_min","y_min","x_max","y_max"
    # ]
    df1 = pd.read_csv(df_path1)
    df2 = pd.read_csv(df_path2)
    common_columns = df1.columns.intersection(df2.columns)[1:]
    merged_df = pd.concat([df1[common_columns], df2[common_columns]])
    merged_df = merged_df.reset_index(drop=True)
    merged_df.to_csv(out_path)
    return merged_df



if __name__ == '__main__':
    # print("------------------test time------------------------")
    excel_filepath = "/data/lijingwen/DDSM/CBIS-DDSM-All-doiJNLP-zzWs5zfZ/Mass-Test-Description.csv"
    out_dirpath = "/data/lijingwen/preprocess_DDSM"
    output_csv_path = '/data/lijingwen/preprocess_DDSM/02_PROCESSED/Test/annos_INITIAL.csv'

    db = Datasets(xlsx_io=excel_filepath, out_path=out_dirpath, mode="test")
    if len(db.df) <= 0:
        raise ValueError('Excel not valid. Please check log error files generated for more information.')
    # 'Converting images to png format'
    db.convert_images_format()
    db.preprocess_image()
    db.df.to_csv(output_csv_path, index=False)
    print(len(db.df))
    #
    map_atts(output_csv_path)
    json_path = '/data/lijingwen/preprocess_DDSM/02_PROCESSED/Test/label_INITIAL.json'
    gen_json(output_csv_path,json_path)


    # print("------------------train time------------------------")
    excel_filepath = "/data/lijingwen/DDSM/CBIS-DDSM-All-doiJNLP-zzWs5zfZ/Mass-Training-Description.csv"
    out_dirpath = "/data/lijingwen/preprocess_DDSM"
    output_csv_path = '/data/lijingwen/preprocess_DDSM/02_PROCESSED/Train/annos_INITIAL.csv'


    db = Datasets(xlsx_io=excel_filepath, out_path=out_dirpath, mode="train")
    if len(db.df) <= 0:
        raise ValueError('Excel not valid. Please check log error files generated for more information.')
    # 'Converting images to png format'
    db.convert_images_format()
    db.preprocess_image()
    db.df.to_csv(output_csv_path, index=False)

    map_atts(output_csv_path)
    json_path = '/data/lijingwen/preprocess_DDSM/02_PROCESSED/Train/label_INITIAL.json'
    gen_json(output_csv_path, json_path)

