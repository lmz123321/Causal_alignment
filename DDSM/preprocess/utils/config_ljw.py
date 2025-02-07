import cv2

IMG_SHAPE: tuple = (350, 350)
PATCH_SIZE: tuple = (300, 300)

LOGGING_DATA_PATH='/data/lijingwen/preprocess_DDSM/LOGS'
CBIS_DDSM_DB_PATH='/data/lijingwen/DDSM/CBIS-DDSM-All-doiJNLP-zzWs5zfZ/CBIS-DDSM'

TEST_CONVERTED_DATA_PATH = '/data/lijingwen/preprocess_DDSM/01_CONVERTED/Test'
TEST_PREPROCESSED_DATA_PATH = '/data/lijingwen/preprocess_DDSM/02_PROCESSED/Test'

TRAIN_CONVERTED_DATA_PATH = '/data/lijingwen/preprocess_DDSM/01_CONVERTED/Train'
TRAIN_PREPROCESSED_DATA_PATH = '/data/lijingwen/preprocess_DDSM/02_PROCESSED/Train'

VAL_CONVERTED_DATA_PATH = '/data/lijingwen/preprocess_DDSM/01_CONVERTED/Val'
VAL_PREPROCESSED_DATA_PATH = '/data/lijingwen/preprocess_DDSM/02_PROCESSED/Val'

PREPROCESSING_CONFIG: str = 'CONF1'
PREPROCESSING_FUNCS: dict = {
    'CONF1': {
        'CROPPING_1': {
            'left': 0.01,
            'right': 0.01,
            'top': 0.04,
            'bottom': 0.04
        },
        'REMOVE_NOISE': {
            'ksize': 3
        },
        'REMOVE_ARTIFACTS': {
            'bin_kwargs': {
                'thresh': 'constant',
                'threshval': 30
            },
            'mask_kwargs': {
                'kernel_shape': cv2.MORPH_ELLIPSE,
                'kernel_size': (20, 10),
                'operations': [(cv2.MORPH_OPEN, None), (cv2.MORPH_DILATE, 2)]
            },
            'contour_kwargs': {
                'convex_contour': False,
            }
        },
        'NORMALIZE_BREAST': {
            'type_norm': 'min_max'
        },
        'FLIP_IMG': {
            'orient': 'left'
        },
        'ECUALIZATION': {
            'clahe_1': {'clip': 2},
        },
        'RATIO_PAD': {
            'ratio': '1:2',
        },
        'RESIZING': {
            'width': IMG_SHAPE[1],
            'height': IMG_SHAPE[0]
        },
        'CROPPING_2': {
            'left': 0.05,
            'right': 0,
            'top': 0,
            'bottom': 0
        },
    },
}




