# utils functions
import time
from ailogger import ailogger

empty_output = [{}]

class Timer:
    test_start_time = None

    def start():
        Timer.test_start_time = time.time()
    
    def tick():
        assert Timer.test_start_time is not None
        return time.time() - Timer.test_start_time

def load_kwargs(kwargs):
    return kwargs["data"], kwargs["model"], kwargs["testconfig"], kwargs["result_path"], kwargs["opt_set"]

def get_saveweight_cb(result_path, save_best_only=True, monitor='loss'):
    import tensorflow as tf
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=result_path,
        verbose=1,
        save_best_only=save_best_only,
        mode='auto',
        save_weights_only=True,
        monitor=monitor,
        period=1
    )

def testcase_func(func):

    def wrap(**kwargs):
        try:
            data, model, testconfig, result_path, opt_set = load_kwargs(kwargs)
            ret = func(data, model, testconfig, result_path, opt_set)
            return ret
        except:
            ailogger.exception("testcase run failed")
            return empty_output
    return wrap


def save_float_img(img,fname):
    import numpy as np
    arr = arr*255
    arr = np.array(arr,dtype=np.uint8)
    img.save(fname)