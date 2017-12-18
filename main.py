
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from layers import architecture, loss_function
from utils.cifar_utils import load_data
import time
from datetime import datetime


# # Data Input

# In[2]:


# Load the raw CIFAR-10 data.
X_train, y_train = load_data(mode='train')
X_test, y_test = load_data(mode='test')


# In[3]:


# Preprocessing: subtract the mean value across every dimension for training data
mean_image = np.mean(X_train, axis=0)
std_image = np.std(X_train, axis=0)

X_train = (X_train.astype(np.float32) - mean_image.astype(np.float32)) / std_image
X_test = (X_test.astype(np.float32) - mean_image) / std_image


# In[4]:


# Reshape it to be size NHWC
X_train = X_train.reshape([X_train.shape[0],3,32,32]).transpose((0,2,3,1))
X_test = X_test.reshape([X_test.shape[0],3,32,32]).transpose((0,2,3,1))

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Test data shape', X_test.shape)
print('Test labels shape', y_test.shape)


# # Declare tensorflow variables and architecture

# In[5]:


X_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
random_rolls = tf.placeholder(dtype=tf.float32, shape=[54])
y_inputs = tf.placeholder(dtype=tf.uint8, shape=[None])
is_training = tf.placeholder(dtype=tf.bool)
y_test_tf = tf.placeholder(dtype=tf.int64, shape=[None])
test_error_tf = tf.placeholder(dtype=tf.float32, shape=())


# In[6]:


train_size = len(X_train)
test_size = len(X_test)
epochs = 500
batch_size = 128
boundaries = [int(.5 * epochs * train_size / batch_size),
              int(.75 * epochs * train_size / batch_size)]
values = [.1, .01, .001]
weight_decay = 1e-4
momentum = 0.9
shuffle_indices = np.arange(train_size)

labels = tf.one_hot(y_inputs, 10)

out = architecture(X_inputs, random_rolls, is_training, strategy='pad',
                   scope='stoch_depth', P=0.5, L=54)

loss = loss_function(logits=out, labels=labels, weight_decay=weight_decay,
                     arch_scope='stoch_depth')
training_summary = tf.summary.scalar('train_loss', loss)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
step = tf.train.MomentumOptimizer(learning_rate,
                                  momentum,
                                  use_nesterov=True).minimize(loss,
                                                              global_step=global_step)
pred = tf.argmax(out, axis=1)
error_num = tf.count_nonzero(pred - y_test_tf, name='error_num')
test_summary = tf.summary.scalar('Test_Error', test_error_tf)


# # Create unique log file name

# In[7]:


now = datetime.now()
datetime_str = now.strftime("%Y%m%d-%H%M%S")
logdir = "tf_logs/{}/".format(datetime_str)


# # Train the network

# In[ ]:


with tf.Session() as sess: 
    
    writer = tf.summary.FileWriter(logdir, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    start_index = 0
    iter_number = -1
    for epoch in range(epochs):
        epoch_start_time = time.time()
        np.random.shuffle(shuffle_indices)
        X_train, y_train = X_train[shuffle_indices], y_train[shuffle_indices]
        start_index = 0
        for _ in range(train_size // batch_size):
            iter_number += 1

            indices = list(range(start_index, start_index + batch_size))
            start_index += batch_size
            X_train_batch, y_train_batch = X_train[indices], y_train[indices]

            # data augmentation
            for inx in range(batch_size):
                x_offset = np.random.randint(-4, 4)
                y_offset = np.random.randint(-4, 4)
                flip_bool = np.random.uniform() > .5
                X_train_batch[inx] = np.roll(X_train_batch[inx],
                                             (x_offset, y_offset),
                                             axis=(0, 1))
                if flip_bool:
                    X_train_batch[inx] = np.flip(X_train_batch[inx], axis=1)

            random_rolls_batch = np.random.uniform(size=54)
            _, loss_val, train_summ = sess.run([step, loss, training_summary],
                                               feed_dict={X_inputs: X_train_batch,
                                                          random_rolls: random_rolls_batch,
                                                          y_inputs: y_train_batch,
                                                          is_training: True})
        writer.add_summary(train_summ, epoch)
        start_index_test = 0
        scores_list = []
        for _ in range(test_size // 100):
            indices_test = list(range(start_index_test, start_index_test + 100))
            start_index_test += 100

            X_test_batch, y_test_batch = X_test[indices_test], y_test[indices_test]

            test_batch_accuracy = sess.run(error_num, feed_dict={X_inputs: X_test_batch,
                                                                 random_rolls: random_rolls_batch,
                                                                 y_test_tf: y_test_batch,
                                                                 is_training: False})
            scores_list.append(test_batch_accuracy)
        test_epoch_accuracy = sum(scores_list) / len(scores_list)
        test_summ = sess.run(test_summary, feed_dict={test_error_tf: test_epoch_accuracy})
        writer.add_summary(test_summ, epoch)

        if epoch % 10 == 0:
            save_path = saver.save(sess, 'checkpoints/{}/model.ckpt'.format(datetime_str))
            print("Model saved in file: %s" % save_path)
        print("epoch_time =", epoch_start_time - time.time())
            
    writer.close()

