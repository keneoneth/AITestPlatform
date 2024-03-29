from ailogger import ailogger
import utils

import os
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
try:
    import Image
except ImportError:
    from PIL import Image


def save_img(arr,fname):
    arr = arr*255
    arr = np.array(arr,dtype=np.uint8)
    im = Image.fromarray(arr)
    im.save(fname+".jpeg")

def batch_normalize_enlarge_imgs(imgs,input_shape,batch_index=0,batch_size=100):
    assert len(input_shape) == 3
    new_imgs = []
    for img in imgs[batch_index*batch_size:(batch_index+1)*batch_size]:
        img = np.true_divide(img, 255) # normalize by 255
        new_imgs.append(np.array(cv2.resize(img, (input_shape[0],input_shape[1]), interpolation = cv2.INTER_LINEAR),dtype=np.float16))
    new_imgs = np.array(new_imgs)
    return new_imgs

def batch_npy(y,batch_index=0,batch_size=100):
    return y[batch_index*batch_size:(batch_index+1)*batch_size]


@utils.testcase_func
def mytest(data, model_key, model, testconfig, result_path, opt_set):
    
    # record test start time
    utils.Timer.start()

    
    # forward model
    num_classes = data.get_class_num()
    if model_key in ["googlenet_model"]:
        model = model.forward(num_classes,tf.keras.Input(shape=tuple(testconfig['input_shape'])))
    elif model_key in ["resnet_model"]:
        model = model.construct(testconfig["resnet_layer"]).forward(10,tf.keras.Input(shape=tuple(testconfig['input_shape'])))
    elif model_key in ["densenet_model"]:
        model = model.construct(testconfig["densenet_layer"]).forward(10,tf.keras.Input(shape=tuple(testconfig['input_shape'])))
    else:
        model = model.forward(num_classes)

    # set loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # set optimizer
    if testconfig['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam()
    elif testconfig['optimizer'] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=testconfig['sgd_learning_rate'], 
            momentum=testconfig['sgd_momentum'], 
            nesterov=False, name='SGD'
        )
    else:
        ailogger.error(f"undefined optimizer {testconfig['optimizer']}")
        raise
    
    # compile model
    model.compile(optimizer=optimizer,loss=loss_fn,metrics=['accuracy'])

    if opt_set.opt_model_path:
        model = tf.keras.models.load_model(opt_set.opt_model_path)

    trial_img_count = 0 # count the output trial img

    # start training model fit
    if opt_set.opt_train:
        for epoch_index in range(testconfig['epochs']):
            ailogger.info("entering epoch %d ..." % epoch_index)
            data.reset() # reset data at start of every epoch
            new_train_data = data.get_train_data()
            while new_train_data != None:
                # split data
                x_train, x_valid, y_train, y_valid = train_test_split(new_train_data['x'], new_train_data['y'], test_size=testconfig['validsize'], random_state=42)

                # fit model batch by batch
                m_batch_size = testconfig['train_batch_size']
                m_batch_num = math.ceil(len(x_train)/m_batch_size)

                for m_batch_index in range(m_batch_num):
                    x_train_batch = batch_normalize_enlarge_imgs(x_train,testconfig['input_shape'],batch_index=m_batch_index,batch_size=m_batch_size)
                    y_train_batch = batch_npy(y_train,batch_index=m_batch_index,batch_size=m_batch_size)

                    if trial_img_count < testconfig["trial_img_no"] and trial_img_count < len(x_train_batch):
                        save_img(
                            x_train_batch[trial_img_count],
                            os.path.join(result_path,f"trial_{trial_img_count}_item_{y_train_batch[trial_img_count]}")
                        )
                        trial_img_count += 1


                    if m_batch_index + 1 < m_batch_num:
                        model.fit(x_train_batch,y_train_batch,batch_size=testconfig['batch_size'],use_multiprocessing=True,workers=os.cpu_count(),verbose=0)
                    else:
                        x_valid = batch_normalize_enlarge_imgs(x_valid,testconfig['input_shape'],batch_size=len(x_valid))
                        model.fit(x_train_batch,y_train_batch,validation_data=(x_valid,y_valid),batch_size=testconfig['batch_size'],use_multiprocessing=True,workers=os.cpu_count(),callbacks=utils.get_saveweight_cb(os.path.join(result_path,'model_ep{epoch:02d}_loss{loss:.2f}.h5')))
                    
                    ### from https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
                    # fit(
                    #     x=None, y=None, batch_size=None, epochs=1, verbose='auto',
                    #     callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,
                    #     class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None,
                    #     validation_steps=None, validation_batch_size=None, validation_freq=1,
                    #     max_queue_size=10, workers=1, use_multiprocessing=False
                    # )

                    x_train_batch = None


                # update new data
                new_train_data = data.get_train_data()

        # save final model
        model.save(os.path.join(result_path, 'final_model.h5'),
                   save_format='h5')

        # print model summary
        ailogger.info(f'model summary {model.summary()}')


    # evaluate model
    if opt_set.opt_test:
        test_data = data.get_test_data()
        test_batch_size = testconfig['test_batch_size']
        test_batch_num = math.ceil(len(test_data['x'])/test_batch_size)
        ret = []
        weights = []
        for test_batch_index in range(test_batch_num):
            test_x_batch = batch_normalize_enlarge_imgs(test_data['x'],testconfig['input_shape'],batch_index=test_batch_index,batch_size=test_batch_size)
            test_y_batch = batch_npy(test_data['y'],batch_index=test_batch_index,batch_size=test_batch_size)
            test_ret = model.evaluate(test_x_batch, test_y_batch, verbose=0)
            ret.append(test_ret[1])
            weights.append(len(test_x_batch))
      
        return [{'avg_acc' : np.average(ret,weights=weights), 'run_time_sec' : float(utils.Timer.tick())}]
    else:
        return utils.empty_output