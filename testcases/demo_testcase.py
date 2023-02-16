from ailogger import ailogger
import utils

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
try:
    import Image
except ImportError:
    from PIL import Image


@utils.testcase_func
def mytest(data, model, testconfig, result_path, opt_set):

    # record test start time
    utils.Timer.start()

    # forward model
    num_classes = data.get_class_num()
    model = model.forward(num_classes)

    # set loss function
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # compile model
    model.compile(optimizer=testconfig['optimizer'],
                  loss=loss_fn, metrics=['accuracy','Precision','Recall'])

    # load input X, Y
    x_train, x_test, y_train, y_test = train_test_split(
        data.get_x(), data.get_y(by_category=True), test_size=testconfig['testsize'], random_state=42)

    ailogger.info(
        f'dataset stats: x_train:{len(x_train)} | y_train:{len(y_train)} | x_test:{len(x_test)} | y_test : {len(y_test)}')

    # load the model if model_path option is activated
    if opt_set.opt_model_path:
        model = tf.keras.models.load_model(opt_set.opt_model_path)

    # fit model if train option is activated
    if opt_set.opt_train:

        model.fit(x_train, y_train, epochs=testconfig['epochs'], callbacks=utils.get_saveweight_cb(
            os.path.join(result_path, 'model_ep{epoch:02d}_loss{loss:.2f}.h5')))

        # save final model
        model.save(os.path.join(result_path, 'final_model.h5'),
                   save_format='h5')

        # print model summary
        ailogger.info(f'model summary {model.summary()}')

    # evaluate model if test option is activated
    if opt_set.opt_test:
        if testconfig['dump_err_img']:
            # add softmax layer
            probability_model = tf.keras.Sequential(
                [model, tf.keras.layers.Softmax()])
            prob_ret = probability_model(x_test)

            # dump error images
            for index, ret in enumerate(prob_ret):
                pred_ret = np.argmax(ret)
                real_ret = y_test[index]
                if np.argmax(ret) != np.argmax(real_ret):
                    two_d_img = np.array(x_test[index].reshape(
                        x_test[index].shape[0:2])*255, np.uint8)
                    img = Image.fromarray(two_d_img)
                    if not os.path.exists(os.path.join(result_path, 'err_img_folder')):
                        os.mkdir(os.path.join(result_path, 'err_img_folder'))
                    img.save(os.path.join(result_path, 'err_img_folder',
                             f'err_idx={index}_real={real_ret}_pred={pred_ret}_img.png'))
            # evaluate model
            loss, accuracy, precision, recall = model.evaluate(
                x_test, y_test, verbose=2)

            return [
                {
                    'loss': loss,
                    'accuracy': accuracy,
                    'f1_score': utils.cal_f1score(precision,recall),
                    'precision': precision,
                    'recall': recall,
                    'run_time_sec': float(utils.Timer.tick())
                }
            ]
        else:
            # evaluate model
            loss, accuracy, precision, recall = model.evaluate(
                x_test, y_test, verbose=2)
            return [
                {
                    'loss': loss,
                    'accuracy': accuracy,
                    'f1_score': utils.cal_f1score(precision,recall),
                    'precision': precision,
                    'recall': recall,
                    'run_time_sec': float(utils.Timer.tick())
                }
            ]
    else:
        return utils.empty_output
