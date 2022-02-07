import os
import time
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
try:
    import Image
except ImportError:
    from PIL import Image

# {b'num_cases_per_batch': 10000, b'label_names': [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck'], b'num_vis': 3072}


def save_img(arr):
    arr = arr*255
    arr = np.array(arr,dtype=np.uint8)
    im = Image.fromarray(arr)
    im.save("trial.jpeg")

def batch_normalize_enlarge_imgs(imgs,input_shape,batch_index=0,batch_size=100):
    assert len(input_shape) == 3
    new_imgs = []
    for img in imgs[batch_index*batch_size:(batch_index+1)*batch_size]:
        img = np.true_divide(img, 255) # normalize by 255
        new_imgs.append(cv2.resize(img, (input_shape[0],input_shape[1]), interpolation = cv2.INTER_LINEAR))
    new_imgs = np.array(new_imgs)
    return new_imgs

def batch_npy(y,batch_index=0,batch_size=100):
    return y[batch_index*batch_size:(batch_index+1)*batch_size]


def mytest(**args):
    
    test_start_time = time.time()

    data = args["data"]
    model = args["model"]
    testfunc = args["testfunc"]
    testconfig = args["testconfig"]
    
    # forward model
    num_classes = 10 #digit 0~9
    model = model.forward(num_classes)
    # set loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # compile model
    model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])

    # model fit
    for epoch_index in range(testconfig['epochs']):
        print("entering epoch %d ..." % epoch_index)
        data.reset()
        new_train_data = data.get_train_data()
        while new_train_data != None:
            # split data
            x_train, x_valid, y_train, y_valid = train_test_split(new_train_data['x'], new_train_data['y'], test_size=testconfig['validsize'], random_state=42)

            


            # fit model batch by batch
            m_batch_size = testconfig['train_batch_size']
            m_batch_num = math.ceil(len(x_train)/m_batch_size)
            # print(m_batch_size,m_batch_num)

            for m_batch_index in range(m_batch_num):
                x_train_batch = batch_normalize_enlarge_imgs(x_train,testconfig['input_shape'],batch_index=m_batch_index,batch_size=m_batch_size)
                y_train_batch = batch_npy(y_train,batch_index=m_batch_index,batch_size=m_batch_size)

                # save_img(x_train_batch[0])
                # print(y_train_batch[0])
                # return [{}]

                # print(x_train_batch.shape,y_train_batch.shape,x_valid.shape,y_valid.shape)
                # save_img(x_train[0])
                if m_batch_index + 1 < m_batch_num:
                    model.fit(x_train_batch,y_train_batch,batch_size=testconfig['batch_size'],use_multiprocessing=True,workers=os.cpu_count(),verbose=0)
                else:
                    x_valid = batch_normalize_enlarge_imgs(x_valid,testconfig['input_shape'],batch_size=len(x_valid))
                    model.fit(x_train_batch,y_train_batch,validation_data=(x_valid,y_valid),batch_size=testconfig['batch_size'],use_multiprocessing=True,workers=os.cpu_count())
                
                ### from https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
                # fit(
                #     x=None, y=None, batch_size=None, epochs=1, verbose='auto',
                #     callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,
                #     class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None,
                #     validation_steps=None, validation_batch_size=None, validation_freq=1,
                #     max_queue_size=10, workers=1, use_multiprocessing=False
                # )

            # update new data
            new_train_data = data.get_train_data()

    # evaluate model
    test_data = data.get_test_data()
    test_batch_size = testconfig['test_batch_size']
    test_batch_num = math.ceil(len(test_data['x'])/test_batch_size)
    ret = []
    for test_batch_index in range(test_batch_num):
        test_x_batch = batch_normalize_enlarge_imgs(test_data['x'],testconfig['input_shape'],batch_index=test_batch_index,batch_size=test_batch_size)
        test_y_batch = batch_npy(test_data['y'],batch_index=test_batch_index,batch_size=test_batch_size)
        test_ret = model.evaluate(test_x_batch, test_y_batch, verbose=0)
        ret.append(test_ret[1])
    # print('ret',ret,len(test_data['x']),test_data['x'].shape)
    
    test_duration = time.time() - test_start_time
    return [{'avg_acc' : np.average(ret), 'run_time_sec' : float(test_duration)}]