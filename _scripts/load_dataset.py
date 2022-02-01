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

def load_normal_image_thread(path, files, format):
    x = []
    for file_name in files:
        if file_name.endswith(format):
            x.append(np.array(Image.open(path + "/" + file_name)))
    return np.array(x)

def load_normal_images(path, format):
    x = None
   
    for root,_,files in os.walk(testcase_path(path)):
        batch_size = math.ceil(len(files)/os.cpu_count())#10000
        batch_num = math.ceil(len(files)/batch_size)
        # print(len(files),batch_num,batch_size,os.cpu_count())
        # for i in range(batch_num):
            # batch_x = load_normal_image_thread(root,files[i*batch_size:(i+1)*batch_size],format)
            # print("bx",batch_x)
            # x = np.concatenate((x,batch_x),axis=0)
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            results = pool.starmap(load_normal_image_thread, [(root,files[i*batch_size:(i+1)*batch_size],format) for i in range(batch_num)] )
            for batch_x in results:
                if x is None:
                    x = batch_x
                else:
                    x = np.concatenate((x,batch_x),axis=0)
    return x
    
def load_npy(path, format):
    fpath = testcase_path(path)+"."+format
    # print(fpath)
    assert os.path.isfile(fpath)
    return np.load(fpath)


class DataLoad:


    load_data_x_map = {
        "jpg" : load_normal_images
    }
    
    load_data_y_map = {
        "npy" : load_npy
    }

    @staticmethod
    def load_dataset(d):
        x = DataLoad.load_data_x_map[d["x_format"]](d["x_name"],d["x_format"])
        y = DataLoad.load_data_y_map[d["y_format"]](d["y_name"],d["y_format"])
        
        return {"x":x,"y":y}