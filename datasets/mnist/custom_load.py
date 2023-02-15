from ailogger import ailogger
from path_config import PathConfig


import os
import numpy as np
import math
import multiprocessing

try:
    import Image
except ImportError:
    from PIL import Image


class MNIST:

    name = 'mnist'
    factor = os.cpu_count() // 2

    # load image matrices to np arrays
    @staticmethod
    def load_normal_image_thread(path, files):
        x = []
        for file_name in files:
            img_arr = np.array(Image.open(os.path.join(path, file_name)))
            if len(img_arr.shape) == 2:
                img_arr = img_arr.reshape(
                    img_arr.shape[0], img_arr.shape[1], 1)
            assert len(img_arr.shape) == 3
            x.append(img_arr)
        return np.array(x) / 255.0

    @staticmethod
    def load_all_images(path):
        x = None

        for root, _, files in os.walk(path):
            batch_size = math.ceil(len(files) / MNIST.factor)
            batch_num = math.ceil(len(files) / batch_size)
            with multiprocessing.Pool(processes=MNIST.factor) as pool:
                results = pool.starmap(MNIST.load_normal_image_thread, [(
                    root, files[i*batch_size:(i+1)*batch_size]) for i in range(batch_num)])
                for batch_x in results:
                    if x is None:
                        x = batch_x
                    else:
                        x = np.concatenate((x, batch_x), axis=0)
        return x

    @staticmethod
    # load image paths only
    def load_image_paths(dir_path):
        ret = []
        for root, _, files in os.walk(dir_path):
            ret = [os.path.join(root, file) for file in files]
        return ret

    @staticmethod
    def get_class_num():
        return 10 # digit 0~9

    @staticmethod
    def get_x():
        x_dpath = PathConfig.get_datasets_path(
            os.path.join(MNIST.name, 'jpg_images'))
        try:
            assert os.path.exists(x_dpath)
        except:
            ailogger.exception(f'file path to input X is missing {x_dpath}')
            raise
        x_arr = MNIST.load_all_images(x_dpath)
        try:
            assert x_arr is not None
        except:
            ailogger.exception(f'input X load failed')
            raise
        return x_arr

    @staticmethod
    def get_y():
        y_fpath = PathConfig.get_datasets_path(
            os.path.join(MNIST.name, 'y_numbers.npy'))
        try:
            assert os.path.exists(y_fpath)
        except:
            ailogger.exception(f'file path to input Y is missing {y_fpath}')
            raise

        return np.load(y_fpath)
