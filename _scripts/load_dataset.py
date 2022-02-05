import os
import cv2
import math
import multiprocessing

try:
    import Image
except ImportError:
    from PIL import Image
import numpy as np


testcase_path = lambda path : "./datasets/"+path

###load image matrices to np arrays
def load_normal_image_thread(path, files):
    x = []
    for file_name in files:
        img_arr = np.array(Image.open(path + "/" + file_name))
        if len(img_arr.shape) == 2:
            img_arr = img_arr.reshape(img_arr.shape[0],img_arr.shape[1],1)
        assert len(img_arr.shape) == 3
        x.append(img_arr)
    return np.array(x)

def load_normal_images(path):
    x = None
   
    for root,_,files in os.walk(testcase_path(path)):
        batch_size = math.ceil(len(files)/os.cpu_count())#10000
        batch_num = math.ceil(len(files)/batch_size)
        # print(len(files),batch_num,batch_size,os.cpu_count())
        # for i in range(batch_num):
            # batch_x = load_normal_image_thread(root,files[i*batch_size:(i+1)*batch_size])
            # print("bx",batch_x)
            # x = np.concatenate((x,batch_x),axis=0)
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            results = pool.starmap(load_normal_image_thread, [(root,files[i*batch_size:(i+1)*batch_size]) for i in range(batch_num)] )
            for batch_x in results:
                if x is None:
                    x = batch_x
                else:
                    x = np.concatenate((x,batch_x),axis=0)
    return x


###load npy data
def load_npy(path):
    fpath = testcase_path(path)+".npy"
    # print(fpath)
    assert os.path.isfile(fpath)
    return np.load(fpath)


class DataLoad:

    load_data_x_map = {
        "load_normal_images" : load_normal_images
    }
    
    load_data_y_map = {
        "load_npy" : load_npy
    }

    @staticmethod
    def load_dataset(dataset_key,d):
        assert d["x_format"] in DataLoad.load_data_x_map
        assert d["y_format"] in DataLoad.load_data_y_map
        x = DataLoad.load_data_x_map[d["x_format"]](dataset_key+'/'+d["x_name"])
        y = DataLoad.load_data_y_map[d["y_format"]](dataset_key+'/'+d["y_name"])
        return {"x":x,"y":y}