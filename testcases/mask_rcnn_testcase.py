import os
from pickletools import optimize
import time
import tensorflow as tf
import numpy as np
try:
    import Image
except ImportError:
    from PIL import Image

import copy

# {b'num_cases_per_batch': 10000, b'label_names': [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck'], b'num_vis': 3072}


def save_img(arr,fname):
    arr = arr*255
    arr = np.array(arr,dtype=np.uint8)
    im = Image.fromarray(arr)
    im.save(fname+".jpeg")

def get_my_metric(metric_name):
    def fn(y_true, y_pred):
        # custom_loss = tf.reduce_mean(y_pred, keepdims=True)
        return y_pred

    fn.__name__ = 'metricname_{}'.format(metric_name)
    return fn

def mytest(**args):
    
    # set test start time
    test_start_time = time.time()

    # unpack arguments
    data = args["data"]()
    model_key = args["model_key"]
    model = args["model"]
    testconfig = args["testconfig"]
    result_path = args["result_path"]
    opt_set = args["opt_set"]
    
    print("[info] model_key",model_key)

    # forward model
    model, anchors, output_len = model.forward(testconfig['num_classes'],tf.keras.Input(shape=tuple(testconfig['input_shape']),dtype=tf.float32),batch_size=testconfig['batch_size'])

    # set loss function
    # set optimizer
    optimizer = tf.keras.optimizers.SGD(
        lr=testconfig['learning_rate'], momentum=testconfig['momentum'],
        clipnorm=testconfig['gradient_clip_norm'])

    # TODO: add model weight
    # Add Losses
    loss_names = [ "rpn_class_loss", "rpn_bbox_loss", "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
    for name in loss_names:
        layer = model.get_layer(name)
        loss = tf.reduce_mean(layer.output, keepdims=True) #* testconfig["LOSS_WEIGHTS"].get(name, 1.)
        model.add_loss(loss)
    
    # Add L2 Regularization
    # Skip gamma and beta weights of batch normalization layers.
    # reg_losses = [
    #     tf.keras.regularizers.l2(testconfig["WEIGHT_DECAY"])(w) / tf.cast(tf.size(w), tf.float32)
    #     for w in model.trainable_weights
    #     if 'gamma' not in w.name and 'beta' not in w.name]
    # model.add_loss(lambda : tf.add_n(reg_losses))

    print("losses",model.losses)

    # compile model
    model.compile(optimizer=optimizer,loss=None,metrics=None)

    #TODO: add metrics
    # Add metrics for losses
    # metrics = [get_my_metric("custom_metric")]
    model.metrics_tensors = []
    for name in loss_names:
        if name in model.metrics_names:
            continue
        layer = model.get_layer(name)
        model.metrics_names.append(name)
        loss = tf.reduce_mean(layer.output, keepdims=True) #* testconfig["LOSS_WEIGHTS"].get(name, 1.)
        # add_metrics?
        model.metrics_tensors.append(lambda : loss)
    print("metrics_tensors",model.metrics_names)

    print(model.summary())
    

    if opt_set.opt_model_path:
        model = tf.keras.models.load_model(opt_set.opt_model_path) #,custom_objects={'AnchorsLayer':anchor_layer})

    # start training model fit
    if opt_set.opt_train:
        # Data generators
        train_generator = data.get_train_data_generator(copy.deepcopy(testconfig),testconfig['batch_size'],copy.deepcopy(anchors))
        valid_generator = data.get_valid_data_generator(copy.deepcopy(testconfig),testconfig['batch_size'],copy.deepcopy(anchors))

        for epoch_index in range(testconfig['epochs']):
            print("entering epoch %d ..." % epoch_index)
            model.fit(x=train_generator,
            validation_data=valid_generator,
            steps_per_epoch=testconfig['steps_per_epoch'],
            validation_steps=testconfig['validation_steps'],
            max_queue_size=100,
            workers=1,
            use_multiprocessing=False)

    # print model summary
    model.save(result_path+'final_model.tf',save_format='tf')

    # evaluate model
    if opt_set.opt_test:
        config = model.get_config() # Returns pretty much every information about your model

        test_generator = data.get_test_data_generator(copy.deepcopy(testconfig),testconfig['batch_size'],copy.deepcopy(anchors))
        test_ret = model.evaluate(x=test_generator, verbose=0)

        test_duration = time.time() - test_start_time

        ### try predict, but seems not working
        # inputs,outputs = test_generator[0]
        # sample_imgs = {
        #     'input_1' : np.array([inputs[0][0]]),
        #     'input_image_meta' : np.array([inputs[1][0]]), 
        #     'input_rpn_match' : np.array([inputs[2][0]]), 
        #     'input_rpn_bbox' : np.array([inputs[3][0]]), 
        #     'input_gt_class_ids' : np.array([inputs[4][0]]), 
        #     'input_gt_boxes' : np.array([inputs[5][0]]), 
        #     'input_gt_masks' : np.array([inputs[6][0]])
        # }
        # for k,v in sample_imgs.items():
        #     print(k,len(v),v)

        # print("ck sample_imgs len",len(sample_imgs))
        # print("ck Generate predictions for some samples")
        # predictions = model.predict(sample_imgs,workers=1,use_multiprocessing=False)
        # print("ck predictions",predictions)
        # print("ck test_ret",test_ret)
        ### try predict, but seems not working

        return [{'avg_loss' : str(test_ret), 'run_time_sec' : float(test_duration)}]
    else:
        return [{}]