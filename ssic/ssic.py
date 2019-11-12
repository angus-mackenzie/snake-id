import os
from ssic import util


# ========================================================================= #
# config                                                                    #
# ========================================================================= #


class SSIC:
    # pretty much only need to change DATASET_DIR or STORAGE_DIR
    STORAGE_DIR          = None
    DATASET_DIR          = None

    # based off of DATASET_DIR
    DATASET_CLASS_CSV   = None
    DATASET_TRAIN_DIR   = None
    DATASET_TEST_DIR    = None

    def __init__(self):
        raise RuntimeError('Cannot instantiate this class')

    @classmethod
    def init(cls):
        cls._init_environ()
        cls._load_vars()

    @classmethod
    def _init_environ(cls):
        # load evironment
        util.load_env()

        # SAVE ORIGINAL or RESTORE TO ORIGINAL
        util.restore_python_path()

        # Methods to visualise CNN activations: https://github.com/utkuozbulak/pytorch-cnn-visualizations
        util.add_python_path('vendor/pytorch-cnn-visualizations/src')
        # Mish activation function: https://github.com/digantamisra98/Mish
        util.add_python_path('vendor/Mish')
        # Variance of the Adaptive Learning Rate: https://github.com/LiyuanLucasLiu/RAdam
        util.add_python_path('vendor/RAdam')
        # Lookahead optimizer: https://github.com/alphadl/lookahead.pytorch
        util.add_python_path('vendor/lookahead.pytorch')
        # Ranger=RAdam+Lookahead: https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
        util.add_python_path('vendor/Ranger-Deep-Learning-Optimizer')

    @classmethod
    def _load_vars(cls):
        # pretty much only need to change DATASET_DIR or STORAGE_DIR
        cls.STORAGE_DIR = util.get_env_path('STORAGE_DIR', 'out')
        cls.DATASET_DIR = util.get_env_path('DATASET_DIR', 'data')
        # based off of DATASET_DIR
        cls.DATASET_CLASS_CSV = util.get_env_path('DATASET_CLASS_CSV', os.path.join(cls.DATASET_DIR, 'class_idx_mapping.csv'))
        cls.DATASET_TRAIN_DIR = util.get_env_path('DATASET_TRAIN_DIR', os.path.join(cls.DATASET_DIR, 'train'))  # path pattern: {DATASET_TRAIN_DIR}/class-{class_id}/{uuid}.{ext}
        cls.DATASET_TEST_DIR = util.get_env_path('DATASET_TEST_DIR',   os.path.join(cls.DATASET_DIR, 'round1'))  # path pattern: {DATASET_TEST_DIR}/{uuid}.{ext}

    @classmethod
    def cache_data(cls, name, func=None):
        def wrapper(_func):
            def inner(*args, **kwargs):
                util.cache_data(
                    path=os.path.join(cls.STORAGE_DIR, name),
                    generator=lambda: _func(*args, **kwargs)
                )
            return inner
        return wrapper if func is None else wrapper(func)

    @classmethod
    def get_class_name_map(cls):
        """
        load all the snake classes, keys are ids, values are names
        """
        import pandas as pd

        mapping = {class_id: name for name, class_id in pd.read_csv(cls.DATASET_CLASS_CSV).values}
        print(f'[\033[92mLOADED\033[0m]: {len(mapping)} classes from: {cls.DATASET_CLASS_CSV}')
        return mapping

    @classmethod
    def get_name_class_map(cls):
        """
        opposite of get_ssic_class_name_map, keys are names, values are ids
        """
        return {name: class_id for (class_id, name) in cls.get_class_name_map().items()}

    @classmethod
    @cache_data('img_info.json')
    def get_train_img_info(cls):
        """
        Get all the paths, names and classes of training images, verifying that images of any paths returned are actually valid.
        """
        from tqdm import tqdm
        from PIL import Image

        info_valid, info_invalid = {}, {}
        # LOOP THROUGH CLASS FOLDERS
        for cls_name in tqdm(os.listdir(cls.DATASET_TRAIN_DIR)):
            cls_path = os.path.join(cls.DATASET_TRAIN_DIR, cls_name)
            cls_id = int(cls_name[len('class-'):])
            # LOOP THROUGH IMAGES IN CLASS FOLDER
            for name in os.listdir(cls_path):
                path, valid = os.path.join(cls_path, name), False
                # make sure we have not seen this before
                assert (name not in info_valid) and (name not in info_invalid), f'Duplicate image name: {path}'
                # validate image
                try:
                    Image.open(path).verify()
                    valid = True
                except (IOError, SyntaxError) as e:
                    pass
                # append data
                (info_valid if valid else info_invalid)[name] = dict(
                    name=name,       # ({uuid}.{ext})
                    path=path,       # ({DATASET_TRAIN_DIR}/class-{id}/{uuid}.{ext})
                    class_id=cls_id  # class-({class_id})
                )

        # Make sure that all classes appear in valid data and vice versa
        classes_csv = set(cls.get_class_name_map())
        classes_img = {info['class_id'] for info in info_valid.values()}
        assert len(classes_csv - classes_img) == 0
        assert len(classes_img - classes_csv) == 0

        print(f'  valid:   {len(info_valid)}')
        print(f'  invalid: {len(info_invalid)}')

        return info_valid, info_invalid

    @classmethod
    @cache_data('human_annotations.json')
    def get_human_annotated_boxes(cls):
        """
        Source article: https://medium.com/@Stormblessed/2460292bcfb
        Data format:
            [{
                class: 'image',
                filename: '{uuid}.{ext}',
                annotations: [{
                    class: 'rect',
                    height: float,
                    width: float,
                    x: float,
                    y: float
                }, ... ]
            }, ... ]
        """
        import json
        import urllib.request

        annotations = json.load(
            urllib.request.urlopen('https://drive.google.com/uc?id=18dx_5Ngmc56fDRZ6YZA_elX-0ehtV5U6')
        )
        print(f'  bounding boxes: {len(annotations)}')
        return annotations


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
