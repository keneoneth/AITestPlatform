# utils functions
import time
from ailogger import ailogger

empty_output = [{}]

class Timer:
    test_start_time = None

    @staticmethod
    def cur():
        return time.time()

    @staticmethod
    def start():
        Timer.test_start_time = Timer.cur()
    
    @staticmethod
    def tick(ref_time=None):
        if ref_time is None:
            assert Timer.test_start_time is not None
            ref_time = Timer.test_start_time
        return time.time() - ref_time
    
def get_timestamp():
    from datetime import datetime  
    str_date_time = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    return str_date_time

def load_kwargs(kwargs):
    return kwargs["data"], kwargs["model_key"],kwargs["model"], kwargs["testconfig"], kwargs["result_path"], kwargs["opt_set"]

def cal_f1score(prec,recall):
    return (2*prec*recall) / (prec+recall)

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
            data, model_key, model, testconfig, result_path, opt_set = load_kwargs(kwargs)
            ret = func(data, model_key, model, testconfig, result_path, opt_set)
            return ret
        except:
            ailogger.exception("testcase run failed")
            return empty_output
    return wrap