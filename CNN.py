# Author     :    Adamya Gupta
# Topic      :    TensorFlow - CNN(Convolutional neural network)
#------------------------------------------------------------------------------------------------------------------------------------------------

import tensorflow as tf 
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt 
from matplotlib import cm 

BATCH_SIZE   =  50
LR = 0.001    # Learning rate 

mnist = input_data.read_data_sets('./mnist', one_hot=True) # They have been normalized to range (0, 1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255. 
image = tf.reshape(tf_x, [-1, 28, 28, 1]) # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, 10]) # input y

# CNN
conv1 = tf.layers.conv2d( # shape (28, 28, 1)
    inputs = image,
    filters= 16,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
    activation=tf.nn.relu
) # (28, 28, 16)
pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2,
)   # (14, 14, 16)

conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)  # (14,14,32)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)  # (7,7,32)
flat = tf.reshape(pool2, [-1, 7*7*32]) # (7*7*32,)
output = tf.layers.dense(flat, 10)    # output layer 


loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)  # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions = tf.argmax(output, axis=1),)[1] 

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)   # Initilize var in graph


for step in range(1000):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    
    accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
    print('Step:', step, '|train loss: %.4f' %loss_, '| test accuracy: %.2f' %accuracy_)


# print 30 predictions from test data
test_output = sess.run(output, {tf_x: test_x[:30]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:30], 1), 'real number')
