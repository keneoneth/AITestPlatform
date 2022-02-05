import sys
sys.path.append("./_scripts/")
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
try:
    import Image
except ImportError:
    from PIL import Image
from load_dataset import load_normal_image_thread


def batch_normalize_enlarge_imgs(img_paths,input_shape,batch_index=0,batch_size=100):
    img_paths = [img_path.numpy().decode('utf-8') for img_path in img_paths]
    imgs = load_normal_image_thread(path=".",files=img_paths[batch_index*batch_size:(batch_index+1)*batch_size])
    assert len(input_shape) == 3
    imgs = np.concatenate(input_shape[2]*[imgs],axis=3)
    new_imgs = []
    for img in imgs[batch_index*batch_size:(batch_index+1)*batch_size]:
        img = np.true_divide(img, 255) # normalize by 255
        new_imgs.append(cv2.resize(img, (input_shape[0],input_shape[1]), interpolation = cv2.INTER_LINEAR))
    new_imgs = np.array(new_imgs)
    return new_imgs

def batch_npy(y,batch_index=0,batch_size=100):
    return y[batch_index*batch_size:(batch_index+1)*batch_size]


def mytest(**args):
    
    data = args["data"]
    model = args["model"]
    testfunc = args["testfunc"]
    testconfig = args["testconfig"]
    
    # compile model
    num_classes = 10 #digit 0~9
    model = model.forward(num_classes)

    # split data
    x_train, x_test, y_train, y_test = train_test_split(data['x'], data['y'], test_size=testconfig['testsize'], random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=testconfig['validsize'], random_state=42)

    batch_size = testconfig['batch_size']
    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    val_dataset = val_dataset.batch(batch_size)

    # Prepare the test dataset.
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)

    # set loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # set optimizer
    optimizer = tf.keras.optimizers.Adam()

    #define acc metrics
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # start training
    for epoch in range(testconfig['epochs']):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            #enlarge and batch data
            batch_x = batch_normalize_enlarge_imgs(x_batch_train,testconfig['input_shape'],batch_index=0)
            batch_y = batch_npy(y_batch_train,batch_index=0)
            # print("ck",batch_x.shape,batch_y.shape)
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
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            #enlarge and batch data
            batch_x = batch_normalize_enlarge_imgs(x_batch_val,testconfig['input_shape'],batch_index=0)
            batch_y = batch_npy(y_batch_val,batch_index=0)

            val_logits = model(batch_x, training=False)
            # Update val metrics
            val_acc_metric.update_state(batch_y, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
    
    # print model summary
    print(model.summary())

    # evaluate model
    for x_batch_test, y_batch_test in test_dataset:
        #enlarge and batch data
        batch_x = batch_normalize_enlarge_imgs(x_batch_test,testconfig['input_shape'],batch_index=0)
        batch_y = batch_npy(y_batch_test,batch_index=0)

        test_logits = model(batch_x, training=False)
        # Update val metrics
        test_acc_metric.update_state(batch_y, test_logits)
    test_acc = test_acc_metric.result()

    return [{'avg_acc' : float(test_acc)}]