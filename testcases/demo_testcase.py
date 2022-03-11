import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import time

def mytest(**args):
    
    test_start_time = time.time()

    data = args["data"]
    model = args["model"]
    testconfig = args["testconfig"]
    result_path = args["result_path"]
    opt_set = args["opt_set"]
    
    
    # forward model
    num_classes = 10 #digit 0~9
    model = model.forward(num_classes)

    # set loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # compile model
    model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])

    x_train, x_test, y_train, y_test = train_test_split(data['x'] / 255.0, data['y'], test_size=testconfig['testsize'], random_state=42)
    print("len(x_train),len(y_train):",len(x_train),len(y_train))
    
    # Create a callback that saves the model's weights every 5 epochs
    # https://medium.com/@italojs/saving-your-weights-for-each-epoch-keras-callbacks-b494d9648202
    
    if opt_set.opt_model_path:
        model = tf.keras.models.load_model(opt_set.opt_model_path)

    # fit model
    if opt_set.opt_train:
        model.fit(x_train, y_train, epochs=testconfig['epochs'],callbacks=opt_set.get_saveweight_cb(result_path+'model_ep{epoch:02d}_loss{loss:.2f}.h5'))

    # print model summary
    print(model.summary())
    model.save(result_path+'final_model.h5',save_format='h5')

    if opt_set.opt_test:
        if testconfig['detailed_comparison']:
            # add softmax layer
            probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
            prob_ret = probability_model(x_test)
            
            # calc accuracy
            correct_count = 0
            for index,ret in enumerate(prob_ret):
                # print(ret,np.argmax(ret),y_test[index])
                if np.argmax(ret) == y_test[index]:
                    correct_count += 1
            test_duration = time.time() - test_start_time
            return [{'avg_acc' : correct_count / len(x_test), 'run_time_sec' : float(test_duration)}]
        else:
            # evaluate model
            ret = model.evaluate(x_test, y_test, verbose=2)
            test_duration = time.time() - test_start_time
            return [{'avg_acc' : ret[1], 'run_time_sec' : float(test_duration)}]
    else:
        return [{}]
    
    

    