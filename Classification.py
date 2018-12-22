# Author :    Adamya Gupta
# Topic :     TensorFlow classification
#-------------------------------------------------------------------------------------------------------------------------

import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 

tf.set_random_seed(1)
np.random.seed(1)



# Fake data

n_data = np.ones((100,2))
x0 = np.random.normal(2*n_data, 1) # class0 x shape=(100,2) , Mean = 2*n_data and standard deviation = 1
y0 = np.zeros(100) # Class0 y shape=(100,1)
x1 = np.random.normal(-2*n_data, 1) # Class1 x shape =(100,2)
y1 = np.ones(100) # Class1 y shape = (100,1)

x = np.vstack((x0,x1)) # shape (200,2) + some noise 
y = np.hstack((y0,y1)) # shape (200, )

# plot data
#plt.scatter(x[:, 0], x[:, 1], c=y, s=100, lw=0, cmap='RdYlGn')
#plt.show()


tf_x = tf.placeholder(tf.float32, x.shape)
tf_y = tf.placeholder(tf.int32, y.shape)


# neural network layer
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)   # hidden layer
output = tf.layers.dense(l1, 2)              # output layer

loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output) # compute cost.

accuracy = tf.metrics.accuracy(labels=tf.squeeze(tf_y), predictions=tf.argmax(output, axis=1),)[1] # return(acc, update_op), and create 2 local variables. 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)  # initilize var in graph

plt.ion()   #something about plotting

for step in range(1000):
    # train and net output
    _, acc, pred = sess.run([train_op, accuracy, output], {tf_x: x, tf_y: y})

    # plot and show learning process
    plt.cla() # Clear the current axis.
    plt.scatter(x[:, 0], x[:, 1], c=pred.argmax(1), s=100, lw=0, cmap='RdYlGn')
    plt.text(1.5, -4, 'Accuracy=%.2f' % acc, fontdict={'size':15, 'color': 'red'})
    plt.pause(0.1)

plt.ioff()
plt.show()

