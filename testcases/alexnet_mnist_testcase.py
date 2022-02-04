import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import cv2


def enlarge_imgs(imgs,input_shape,batch_index=0,batch_size=100):
    assert len(input_shape) == 3
    imgs = np.concatenate(input_shape[2]*[imgs],axis=3)
    new_imgs = []
    for img in imgs[batch_index*batch_size:(batch_index+1)*batch_size]:
        new_imgs.append(cv2.resize(img, (input_shape[0],input_shape[1]), interpolation = cv2.INTER_LINEAR))
    new_imgs = np.array(new_imgs)
    print("ck",new_imgs.shape)
    return new_imgs

def batch_y(answers,batch_index=0,batch_size=100):
    return answers[batch_index*batch_size:(batch_index+1)*batch_size]

def mytest(**args):
    
    data = args["data"]
    model = args["model"]
    testfunc = args["testfunc"]
    testconfig = args["testconfig"]
    
    #enlarge and batch data
    data['x'] = enlarge_imgs(data['x'],testconfig['input_shape'])
    data['y'] = batch_y(data['y'])

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

    # evaluate model
    model.evaluate(x_test, y_test, verbose=2)

    # add softmax layer
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    prob_ret = probability_model(x_test)
    
    # calc accuracy
    correct_count = 0
    for index,ret in enumerate(prob_ret):
        # print(ret,np.argmax(ret),y_test[index])
        if np.argmax(ret) == y_test[index]:
            correct_count += 1
    
    # print model summary
    print(model.summary())

    return [{'avg_acc' : correct_count / len(x_test)}]