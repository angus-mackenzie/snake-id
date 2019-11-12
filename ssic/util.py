import os
import sys
import json
import dotenv


def restore_python_path():
    """
    Saves the original sys.path on the first call, later uses this to reset the sys.path
    """
    # SAVE sys.path
    global _ORIG_SYS_PATH
    if '_ORIG_SYS_PATH' not in globals():
        _ORIG_SYS_PATH = list(sys.path) # shallow copy
    else:
        print('[\033[93mRESTORED\033[0m]: sys.path')
        sys.path = list(_ORIG_SYS_PATH) # shallow copy


def add_python_path(path):
    """
    Appends to sys.path
    """
    sys.path.insert(0, os.path.abspath(path))


def cache_data(path, generator, regen=False):
    """
    This function caches data from the passed function.
    - If the specified path does not exist, then the function is called to generate the data and the data is then saved for future use.
    - If the specified path does exist, the function is ignored and the cached data is immediately loaded.
    """
    assert path.endswith('.json'), 'file name must end with .json'
    if regen or not os.path.exists(path):
        # GENERATE DATA & CACHE
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f'[\033[93mGENERATING\033[0m]: {path}')
        data = generator()
        with open(path, 'w') as file:
            json.dump(data, file)
        print(f'[\033[92mSAVED\033[0m]: {path}')
    else:
        # LOAD DATA IF EXISTS
        with open(path, 'r') as file:
            data = json.load(file)
        print(f'[\033[92mLOADED\033[0m]: {path}')
    return data


def load_env():
    """
    Saves the original os.environ on the first call, later uses this to reset the os.environ
    """
    
    # SAVE os.environ
    global _ORIG_ENVIRON
    if '_ORIG_ENVIRON' not in globals():
        # this might break things... os.environ is a specific class that extends a dict.
        _ORIG_ENVIRON = dict(os.environ)
    else:
        print('[\033[93mRESTORED\033[0m]: os.environ')
        os.environ = dict(_ORIG_ENVIRON)
        
    # LOAD .env FILE
    if dotenv.load_dotenv(dotenv.find_dotenv()):
        print(f'[\033[92mLOADED\033[0m]: {dotenv.find_dotenv()}')
    else:
        print(f'[\033[91mWARNING\033[0m]: no .env file found!')
        

def get_env_path(key, default):
    """
    Same as os.environ.get, but converts
    paths to their absolute representation.
    """
    path = os.environ.get(key, default)
    path = os.path.abspath(path)
    print(f'[\033[92m{key}\033[0m]: \033[90m{path}\033[0m')
    return path

def set_random_seed(seed=42):
    """
    Set the random seed for reproducability
    https://docs.fast.ai/dev/test.html#getting-reproducible-results
    """
    
    assert isinstance(seed, int)    
    
    # python RNG
    import random
    random.seed(seed)

    # pytorch RNGs
    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

    # numpy RNG
    import numpy as np
    np.random.seed(seed)
    
    print(f'[SEEDED]: {seed}')