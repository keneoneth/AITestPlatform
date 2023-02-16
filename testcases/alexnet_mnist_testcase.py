from ailogger import ailogger
import utils

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def enlarge_imgs(imgs,shape):
    ret = []
    for img in imgs:
        enlarged_img = cv2.resize(img.numpy(), (shape[0],shape[1]), interpolation = cv2.INTER_LINEAR)
        enlarged_img = enlarged_img.reshape(shape[0],shape[1],1)
        ret.append(enlarged_img)
    return ret


@utils.testcase_func
def alexnet_mnist_testcase(data, model, testconfig, result_path, opt_set):
    
    # record test start time
    utils.Timer.start()

    # forward model
    num_classes = data.get_class_num()
    model = model.forward(num_classes)

    # set loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # set optimizer
    if testconfig['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam()
    else:
        ailogger.error(f"undefined optimizer {testconfig['optimizer']}")
        raise

    # split data to train, test, valid set
    x_train, x_test, y_train, y_test = train_test_split(data.get_x(), data.get_y(), test_size=testconfig['testsize'], random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=testconfig['validsize'], random_state=42)

    batch_size = testconfig['batch_size']
    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size)

    # Prepare the test dataset.
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    ailogger.info(
        f'dataset stats: batch_train_set:{len(train_dataset)} | batch_valid_set:{len(val_dataset)} | batch_test_set : {len(test_dataset)}')

    

    #define acc metrics
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    if opt_set.opt_model_path:
        model = tf.keras.models.load_model(opt_set.opt_model_path)

    # start training
    if opt_set.opt_train:
        # iterate epoch by epoch
        for epoch in range(testconfig['epochs']):
            ailogger.info("Start of epoch %d" % epoch)

            epoch_start_time = utils.Timer.cur()

            # Iterate over the batches of the dataset.
            for step, (batch_x, batch_y) in enumerate(train_dataset):

                batch_x = tf.convert_to_tensor(enlarge_imgs(batch_x,testconfig['input_shape']))

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:
                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits = model(batch_x, training=True)  # Logits for this minibatch
                    # Compute the loss value for this minibatch.
                    loss_value = loss_fn(batch_y, logits)
                # Use the gradient tape to automatically retrieve

                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Log every 200 batches.
                if step % 200 == 0:
                    ailogger.info(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    ailogger.info("Seen so far: %s samples" % ((step + 1) * batch_size))
                    model.save(result_path+'model_ep{epoch:02d}_loss{loss:.2f}.h5'.format(epoch=step, loss=float(loss_value)),save_format='h5')

            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            ailogger.info("Training acc over epoch: %.4f" % float(train_acc))

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for batch_x, batch_y in val_dataset:
                batch_x = tf.convert_to_tensor(enlarge_imgs(batch_x,testconfig['input_shape']))
                val_logits = model(batch_x, training=False)
                # Update val metrics
                val_acc_metric.update_state(batch_y, val_logits)

            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            ailogger.info("Validation acc over epoch: %.4f" % float(val_acc))
            ailogger.info("Time taken: %.2fs" % utils.Timer.tick(epoch_start_time))

        # save final model
        model.save(result_path+'final_model.h5',save_format='h5')

        # print model summary
        ailogger.info(f'model summary {model.summary()}')
        

    # evaluate model
    if opt_set.opt_test:
        for batch_x, batch_y in test_dataset:
            batch_x = tf.convert_to_tensor(enlarge_imgs(batch_x,testconfig['input_shape']))
            test_logits = model(batch_x, training=False)
            # Update val metrics
            test_acc_metric.update_state(batch_y, test_logits)
        test_acc = test_acc_metric.result()

        test_duration = utils.Timer.tick()

        return [{'avg_acc' : float(test_acc),'run_time_sec' : float(test_duration)}]
    else:
        return utils.empty_output