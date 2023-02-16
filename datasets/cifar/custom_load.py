import os
import pickle
import numpy as np
from path_config import PathConfig




def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

class load_cifar10():
    '''
    label names of cifar-10
    [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck']
    '''
    
    name = 'cifar'
    testcase_path = lambda path : PathConfig.get_datasets_path(os.path.join(load_cifar10.name,"cifar-10-batches-py/",path))
    data_fnames = ["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5"]
    test_fname = "test_batch"
    data_index = 0

    @staticmethod
    def reset():
        load_cifar10.data_index = 0

    @staticmethod
    def get_train_data():
        if load_cifar10.data_index < len(load_cifar10.data_fnames):
            ret = unpickle(load_cifar10.testcase_path(load_cifar10.data_fnames[load_cifar10.data_index]))
            load_cifar10.data_index += 1
            ret = {'x': np.array(ret[str.encode('data')],dtype=np.uint8), 'y': np.array(ret[str.encode('labels')])}
            ret['x'] = np.transpose(ret['x'].reshape((-1,3,32,32)), (0, 2, 3, 1))
            return ret
        return None
        
    @staticmethod
    def get_test_data():
        ret = unpickle(load_cifar10.testcase_path(load_cifar10.test_fname))
        ret = {'x': np.array(ret[str.encode('data')],dtype=np.uint8), 'y': np.array(ret[str.encode('labels')])}
        ret['x'] = np.transpose(ret['x'].reshape((-1,3,32,32)), (0, 2, 3, 1))
        return ret