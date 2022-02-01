import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

def mytest(**args):
    
    data = args["data"]
    model = args["model"]
    testfunc = args["testfunc"]
    testconfig = args["testconfig"]
    
    
    x_train, x_test, y_train, y_test = train_test_split(data['x'] / 255.0, data['y'], test_size=testconfig['testsize'], random_state=42)
    print("len(x_train),len(y_train):",len(x_train),len(y_train))
    
    # fit model
    model.fit(x_train, y_train, epochs=5)

    # evaluate model
    model.evaluate(x_test, y_test, verbose=2)

    # add softmax layer
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    prob_ret = probability_model(x_test)
    
    correct_count = 0
    for index,ret in enumerate(prob_ret):
        # print(ret,np.argmax(ret),y_test[index])
        if np.argmax(ret) == y_test[index]:
            correct_count += 1
    
    # print(x_test[0])
    # print(x_test[1])
    
    return {'avg_acc' : correct_count / len(x_test)}