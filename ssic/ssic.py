import os
from ssic import util

# ========================================================================= #
# util                                                                      #
# ========================================================================= #


def _cache(name):
    # decorator that does the same thing as util.cache_data, but uses cls.STORAGE_DIR as the default base.
    def wrapper(_func):
        def inner(cls, *args, **kwargs):
            return util.cache_data(
                path=os.path.join(cls.STORAGE_DIR, name),
                generator=lambda: _func(cls, *args, **kwargs)
            )
        return inner
    return wrapper


# ========================================================================= #
# config                                                                    #
# ========================================================================= #


class __SSIC:

    def __init__(self):
        # pretty much only need to change DATASET_DIR or STORAGE_DIR
        self.STORAGE_DIR       = None
        self.DATASET_DIR       = None
        # based off of DATASET_DIR
        self.DATASET_CLASS_CSV = None
        self.DATASET_TRAIN_DIR = None
        self.DATASET_TEST_DIR  = None
        # classes
        self._class_name_map   = None
        self._name_class_map   = None

    def init(self):
        self._init_environ()
        self._load_vars()

    def _init_environ(self):
        # load evironment
        util.load_env()

        # SAVE ORIGINAL or RESTORE TO ORIGINAL
        util.restore_python_path()

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Methods to visualise CNN activations: https://github.com/utkuozbulak/pytorch-cnn-visualizations
        util.add_python_path(f'{root_dir}/vendor/pytorch-cnn-visualizations/src')
        # Mish activation function: https://github.com/digantamisra98/Mish
        util.add_python_path(f'{root_dir}/vendor/Mish')
        # Variance of the Adaptive Learning Rate: https://github.com/LiyuanLucasLiu/RAdam
        util.add_python_path(f'{root_dir}/vendor/RAdam')
        # Lookahead optimizer: https://github.com/alphadl/lookahead.pytorch
        util.add_python_path(f'{root_dir}/vendor/lookahead.pytorch')
        # Ranger=RAdam+Lookahead: https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
        util.add_python_path(f'{root_dir}/vendor/Ranger-Deep-Learning-Optimizer')

    def _load_vars(self):
        # pretty much only need to change DATASET_DIR or STORAGE_DIR
        self.STORAGE_DIR = util.get_env_path('STORAGE_DIR', 'out')
        self.DATASET_DIR = util.get_env_path('DATASET_DIR', 'data')
        # based off of DATASET_DIR
        self.DATASET_CLASS_CSV = util.get_env_path('DATASET_CLASS_CSV', os.path.join(self.DATASET_DIR, 'class_idx_mapping.csv'))
        self.DATASET_TRAIN_DIR = util.get_env_path('DATASET_TRAIN_DIR', os.path.join(self.DATASET_DIR, 'train'))  # path pattern: {DATASET_TRAIN_DIR}/class-{class_id}/{uuid}.{ext}
        self.DATASET_TEST_DIR  = util.get_env_path('DATASET_TEST_DIR',  os.path.join(self.DATASET_DIR, 'round1'))  # path pattern: {DATASET_TEST_DIR}/{uuid}.{ext}

    @property
    def class_name_map(self):
        """
        load all the snake classes, keys are ids, values are names
        """
        if self._class_name_map is None:
            import pandas as pd
            self._class_name_map = {class_id: name for name, class_id in pd.read_csv(self.DATASET_CLASS_CSV).values}
            print(f'[\033[92mLOADED\033[0m]: {len(self._class_name_map)} classes from: {self.DATASET_CLASS_CSV}')
        return self._class_name_map

    def name_class_map(self):
        """
        opposite of get_ssic_class_name_map, keys are names, values are ids
        """
        if self._name_class_map is None:
            self._name_class_map = {name: class_id for (class_id, name) in self.class_name_map.items()}
        return self._name_class_map
    
    @property
    def num_classes(self):
        return len(self.class_name_map)

    @_cache('img_info.json')
    def get_train_image_info(self):
        """
        Get all the paths, names and classes of training images, verifying that images of any paths returned are actually valid.
        """
        from tqdm import tqdm
        from PIL import Image

        info = {}
        # LOOP THROUGH CLASS FOLDERS
        for cls_name in tqdm(os.listdir(self.DATASET_TRAIN_DIR)):
            cls_path = os.path.join(self.DATASET_TRAIN_DIR, cls_name)
            cls_id = int(cls_name[len('class-'):])
            # LOOP THROUGH IMAGES IN CLASS FOLDER
            for name in os.listdir(cls_path):
                path, valid = os.path.join(cls_path, name), False
                # make sure we have not seen this before
                assert name not in info, f'Duplicate image name: {path}'
                # validate image
                try:
                    Image.open(path).verify()
                    valid = True
                except (IOError, SyntaxError) as e:
                    pass
                # append data
                info[name] = dict(
                    name=name,       # ({uuid}.{ext})
                    path=path,       # ({DATASET_TRAIN_DIR}/class-{id}/{uuid}.{ext})
                    class_id=cls_id, # class-({class_id})
                    valid=valid
                )

        # Make sure that all classes appear in valid data and vice versa
        classes_csv = set(self.class_name_map)
        classes_img = {info['class_id'] for info in info.values()}
        assert len(classes_csv - classes_img) == 0
        assert len(classes_img - classes_csv) == 0

        print(f'  valid:   {sum(inf["valid"] for inf in info.values())}')
        print(f'  invalid: {sum(not inf["valid"] for inf in info.values())}')

        return info
    
    def get_random_img_info(self):
        import random
        image_info = self.get_train_image_info()
        return image_info[random.choice(list(image_info))]
    
    def get_random_img_data(self):
        from PIL import Image
        import numpy as np
        info = self.get_random_img_info()
        return np.array(Image.open(info['path'])), info
    
    def get_random_img(self):
        return self.get_random_img_data()[0]

    def get_train_imagelist(self, validate_ratio=0.2):
        from fastai.vision import ImageList
        return ImageList([
            info['path'] for info in self.get_train_image_info().values() if info['valid']
        ]).split_by_rand_pct(validate_ratio).label_from_folder()

    @_cache('human_annotations.json')
    def get_human_annotated_boxes(self):
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

# INIT
SSIC = __SSIC()

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
