import tensorflow as tf

# build model
mymodel = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# set loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# compile model
mymodel.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])

#print summary
print(mymodel.summary())