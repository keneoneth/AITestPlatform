import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

def mytest(**args):
    
    data = args["data"]
    model = args["model"]
    testfunc = args["testfunc"]
    testconfig = args["testconfig"]
    
    # forward model
    num_classes = 10 #digit 0~9
    model = model.forward(10)

    # set loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # compile model
    model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])

    x_train, x_test, y_train, y_test = train_test_split(data['x'] / 255.0, data['y'], test_size=testconfig['testsize'], random_state=42)
    print("len(x_train),len(y_train):",len(x_train),len(y_train))
    
    # fit model
    model.fit(x_train, y_train, epochs=testconfig['epochs'])

    # print model summary
    print(model.summary())


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
        return [{'avg_acc' : correct_count / len(x_test)}]
    else:
        # evaluate model
        ret = model.evaluate(x_test, y_test, verbose=2)
        return [{'avg_acc' : ret[1]}]
    
    

    