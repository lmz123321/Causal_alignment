"image file path"
import json
import os
import sys
sys.path.append('/home/lijingwen/Projects/Counter_align/baseline/breast_cancer_diagnosis-master/src')
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
#
# from preprocessing.image_processing import full_image_pipeline


def jitter(box, jittersize):
    #box:xmax ymax xmin ymin

    jity = np.random.randint(low=-jittersize[1], high=jittersize[1] + 1)
    jitx = np.random.randint(low=-jittersize[0], high=jittersize[0] + 1)

    return [box[0] + jitx, box[1] + jity, box[2] + jitx, box[3] + jity]

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

def cal_mask_boundaries(mask):

    # 找到 mask 图像的上下左右边界
    non_zero_rows = cv2.findNonZero(mask)
    if non_zero_rows is not None:
        x_min, y_min = non_zero_rows.min(axis=0)[0]
        x_max, y_max = non_zero_rows.max(axis=0)[0]

    else:
        # 如果 mask 图像全为零，设置默认值或根据需求处理
        x_min, y_min = 0, 0
        x_max, y_max = 0, 0

    return x_min, y_min, x_max, y_max


def gen_JIT_img(processed_dir,converted_dir,ddsm_db_path,split,jit=(20,20),pad=(30,30)):
    '''
    :param processed_dir: /data/lijingwen/preprocess_DDSM/02_PROCESSED
    :param converted_dir: /data/lijingwen/preprocess_DDSM/01_CONVERTED(dcm转npy的图像存放其中)
    :param split:
    :return:
    '''
    jit = (200, 200)
    pad = (250, 250)
    os.makedirs(os.path.join(processed_dir,split,'UNRESIZE'), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, split, 'BIG-JIT_out'), exist_ok=True)  # 创建precessed UNRESIZE图像文件夹
    os.makedirs(os.path.join(processed_dir, split, 'BIG-JIT'), exist_ok=True)#创建precessed JIT图像文件夹
    os.makedirs(os.path.join(processed_dir, split, 'UNRESIZE-MASK'), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, split, 'BIG-JIT-MASK'), exist_ok=True)
    csv_path=os.path.join(processed_dir,split,'annos.csv')
    df_resize=pd.read_csv(csv_path)

    new_attributes = ['jit_xmin','jit_ymin','jit_xmax','jit_ymax','jmask_xmin', 'jmask_ymin','jmask_xmax', 'jmask_ymax']#df新增属性,jit_xmin jit_box在unresize上坐标，jmask_xmin是jit上的mask坐标
    for attribute in new_attributes:
        df_resize[attribute] = 0#新增属性置0
    exception_ids=[]
    exc_pipeline_ids = []
    for index, row in tqdm(df_resize.iterrows(), total=len(df_resize)):
        id=row['id']

        img_filepath = os.path.join(converted_dir,split,'FULL',f'{id}.npy')  # 读取01_CONVERTED中的img npy
        dest_filepath = os.path.join(processed_dir,split,'UNRESIZE',f'{id}.npy')

        img_mask_filepath = os.path.join(ddsm_db_path,row['ROI mask file path'])
        img_mask_out_path = os.path.join(processed_dir,split,'UNRESIZE-MASK',f'{id}.npy')

        jit_img_savepath = os.path.join(processed_dir, split, 'BIG-JIT', f'{id}.npy')
        jit_mask_savepath = os.path.join(processed_dir, split, 'BIG-JIT-MASK', f'{id}.npy')

        # #获取预处理后，未RESIZE的IMG和MASK,存储到UNRESIZE文件夹中
        # if_exception = full_image_pipeline(img_filepath, dest_filepath, img_mask_filepath, img_mask_out_path,if_resize=False)
        # if if_exception is True:
        #     exc_pipeline_ids.append(id)
        #     continue

        #进行jit
        #先获取mask坐标
        # img = cv2.resize(np.load(dest_filepath),(224,224),interpolation=cv2.INTER_NEAREST)
        # mask = cv2.resize(np.load(img_mask_out_path),(224,224),interpolation=cv2.INTER_NEAREST)
        img =np.load(dest_filepath)
        mask =np.load(img_mask_out_path)
        h,w=img.shape


        xmin,ymin,xmax,ymax =cal_mask_boundaries(mask)#[xmin,ymin,xmax,ymax]
        pad_box=[xmax+pad[0],ymax+pad[1],xmin-pad[0],ymin-pad[1]]

        jit_box=jitter(pad_box, jit)#output jit coords:xmax ymax xmin ymin
        count=0
        while jit_box[0]>=w or jit_box[1]>=h or jit_box[2]<0 or jit_box[3]<0:
            jit_box = jitter(pad_box, jit)  # output jit coords:xmax ymax xmin ymin
            count+=1
            if count>500:
                #到边界的由另外一边补齐
                if jit_box[0]>w:
                    jit_box[2]=max(jit_box[2]-(jit_box[0]-w),0)
                    jit_box[0]=w-1
                elif jit_box[2]<0:
                    jit_box[0]=min(jit_box[0]+(0-jit_box[2]),w-1)
                    jit_box[2]=0
                if jit_box[1] > h:
                    jit_box[3] = max(jit_box[3] - (jit_box[1] - h),0)
                    jit_box[1]=h-1
                elif jit_box[3]<0:
                    jit_box[1] = min(jit_box[1] + (0-jit_box[3]),h-1)
                    jit_box[3]=0
                # jit_box[0]=min(jit_box[0],w)
                # jit_box[1] = min(jit_box[1], h)
                # jit_box[2] = max(jit_box[2], 0)
                # jit_box[3] = max(jit_box[3], 0)
                exception_ids.append(id)
                break

        jit_img=img[jit_box[3]:jit_box[1],jit_box[2]:jit_box[0]]
        jit_mask=mask[jit_box[3]:jit_box[1], jit_box[2]:jit_box[0]]
        np.save(jit_img_savepath,jit_img)
        np.save(jit_mask_savepath,jit_mask)

        #记录jit box在原图上的坐标，存储到csv中
        df_resize.at[index,'jit_xmax'],df_resize.at[index,'jit_ymax'],df_resize.at[index,'jit_xmin'],df_resize.at[index,'jit_ymin']=jit_box[0],jit_box[1],jit_box[2],jit_box[3]
        # row['jit_xmax'],row['jit_ymax'],row['jit_xmin'],row['jit_ymin']=jit_box[0],jit_box[1],jit_box[2],jit_box[3]
        #记录jit_img中mask坐标，存储到csv中
        df_resize.at[index,'jmask_xmin'], df_resize.at[index,'jmask_ymin'], df_resize.at[index,'jmask_xmax'], df_resize.at[index,'jmask_ymax']=cal_mask_boundaries(jit_mask)
        # row['jmask_xmin'], row['jmask_ymin'], row['jmask_xmax'],row['jmask_ymax'] = cal_mask_boundaries(jit_mask)

        # # 加载npy文件
        # # img = np.load(dest_filepath)
        # # mask = np.load(img_mask_out_path)
        # jit_img = np.load(jit_img_savepath)
        # jit_mask = np.load(jit_mask_savepath)


        if count>=500:
            # 创建画布和子图
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            # 第一张子图
            axs[0].imshow(img)
            axs[0].imshow(mask, alpha=0.5)  # 绘制透明mask
            axs[0].axvline(x=df_resize.at[index,'jit_xmin'], color='r', linestyle='--', linewidth=2)  # 绘制jit坐标线
            axs[0].axvline(x=df_resize.at[index,'jit_xmax'], color='r', linestyle='--', linewidth=2)
            axs[0].axhline(y=df_resize.at[index,'jit_ymin'], color='r', linestyle='--', linewidth=2)
            axs[0].axhline(y=df_resize.at[index,'jit_ymax'], color='r', linestyle='--', linewidth=2)
            axs[0].set_title('Original Image with Mask and Jit Box')

            # 第二张子图
            axs[1].imshow(jit_img)
            axs[1].imshow(jit_mask, alpha=0.5)  # 绘制透明jit_mask
            axs[1].axvline(x=df_resize.at[index,'jmask_xmin'], color='r', linestyle='--', linewidth=2)  # 绘制jit_mask坐标线
            axs[1].axvline(x=df_resize.at[index,'jmask_xmax'], color='r', linestyle='--', linewidth=2)
            axs[1].axhline(y=df_resize.at[index,'jmask_ymin'], color='r', linestyle='--', linewidth=2)
            axs[1].axhline(y=df_resize.at[index,'jmask_ymax'], color='r', linestyle='--', linewidth=2)
            axs[1].set_title('Jit Image with Jit Mask')
            # plt.show()
            plt.savefig(os.path.join(processed_dir, split, 'BIG-JIT_out',f"{split}_{id}.png"))
            # print("a")


    df_resize.to_csv(os.path.join(processed_dir, split,'annos_bigjit.csv'),index=False)

    print(f'共{len(exc_pipeline_ids)}个npy pipeline转化存在问题 {exc_pipeline_ids} ')
    print(f'共{len(exception_ids)}个npy jit转化存在问题 {exception_ids} ')

def df2json(csv_path,json_path,columns):

    df = pd.read_csv(csv_path)

    # 提取 'id' 和 'pathology' 两列
    df_subset = df[columns]

    # 转换为 JSON 格式
    result_dict = {'data': df_subset.to_dict(orient='records')}
    result_json = json.dumps(result_dict, ensure_ascii=False, indent=2)

    with open(json_path, 'w') as f:
        f.write(result_json)

    print(f"DataFrame subset has been saved to {json_path}")


if __name__ == '__main__':
    processed_dir="/data/lijingwen/preprocess_DDSM/02_PROCESSED"
    converted_dir="/data/lijingwen/preprocess_DDSM/01_CONVERTED"
    ddsm_db_path='/data/lijingwen/DDSM/CBIS-DDSM-All-doiJNLP-zzWs5zfZ/CBIS-DDSM'
    for split in ['Train','Val']:
        gen_JIT_img(processed_dir,converted_dir,ddsm_db_path,split)

    for split in ['Train', 'Val','Test']:
        csv_path=os.path.join(processed_dir,split,'annos_112bigjit.csv')
        json_path = os.path.join(processed_dir, split, 'annos_112bigjit.json')
        columns=['id','pathology']
        df2json(csv_path, json_path, columns)


