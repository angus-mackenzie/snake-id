import os


class folders():
    # pretty much only need to change DATASET_DIR or STORAGE_DIR
    DATASET_DIR          = util.get_env_path('DATASET_DIR', 'data')
    STORAGE_DIR          = util.get_env_path('STORAGE_DIR', 'out')
    # based off of DATASET_DIR
    DATASET_SSIC_CLASSES = util.get_env_path('DATASET_SSIC_CLASSES', os.path.join(DATASET_DIR, 'class_idx_mapping.csv'))
    DATASET_SSIC_TRAIN   = util.get_env_path('DATASET_SSIC_TRAIN', os.path.join(DATASET_DIR, 'train'))  # path pattern: {DATASET_SSIC_TRAIN}/class-{class_id}/{uuid}.{ext}
    DATASET_SSIC_TEST    = util.get_env_path('DATASET_SSIC_TEST', os.path.join(DATASET_DIR, 'round1'))  # path pattern: {DATASET_SSIC_TEST}/{uuid}.{ext}



def get_folder_image_info(root):
    """
    List all the files in the spcified folder.
    labels files whether or not they are valid images.
    """
    from PIL import Image

    info = []
    
    # loop through all files in folder
    for name in os.listdir(root):
        path, valid = os.path.join(root, name), False
        try:
            Image.open(path).verify()
            valid = True
        except (IOError, SyntaxError) as e:
            pass
        info.append((name, path, valid))
    return info


def get_ssic_class_name_map(ssic_class_csv):
    """
    load all the snake classes
    """

    import pandas as pd
    
    mapping = {class_id: name for name, class_id in pd.read_csv(ssic_class_csv).values}
    print(f'[\033[92mLOADED\033[0m]: {len(mapping)} classes from: {ssic_class_csv}')

    return mapping

def get_ssic_name_class_map(ssic_class_csv):
    return {name: class_id for (class_id, name) in get_ssic_class_name_map(ssic_class_csv).items()}


def get_ssic_train_img_info(ssic_train_folder):
    """
    Get all the paths, names and classes of training images, verifying that images of any paths returned are actually valid.
    """
    from tqdm import tqdm
    
    info_valid, info_invalid = {}, {}
    
    # LOOP THROUGH CLASS FOLDERS
    for cls_name in tqdm(os.listdir(ssic_train_folder)):
        cls_path = os.path.join(ssic_train_folder, cls_name)
        cls_id = int(cls_name[len('class-'):])
        # LOOP THROUGH IMAGES IN CLASS FOLDER
        for (name, path, valid) in get_folder_image_info(cls_path):
            assert (name not in info_valid) and (name not in info_invalid), f'Duplicate image name: {path}'
            (info_valid if valid else info_invalid)[name] = dict(
                name=name,      # ({uuid}.{ext})
                path=path,      # ({DATASET_SSIC_TRAIN}/class-{id}/{uuid}.{ext})
                class_id=cls_id # class-({class_id})
            )
            
    return info_valid, info_invalid

