
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from layers import architecture, evaluate
from utils.cifar_utils import load_data


# In[2]:


# Load the raw CIFAR-10 data.
X_train, y_train = load_data(mode='train')
X_test, y_test = load_data(mode='test')

# Data organizations:
# Train data: 49000 samples from original train set: 1~49000
# Validation data: 1000 samples from original train set: 49000~50000
num_training = 45000
num_validation = 5000

X_val = X_train[-num_validation:, :]
y_val = y_train[-num_validation:]

X_train = X_train[:num_training, :]
y_train = y_train[:num_training]


# In[3]:


# # Preprocessing: subtract the mean value across every dimension for training data, and reshape it to be RGB size
# mean_image = np.mean(X_train, axis=0)
# std_image = np.std(X_train, axis=0)

# X_train = (X_train.astype(np.float32) - mean_image.astype(np.float32)) / std_image
# X_val = (X_val.astype(np.float32) - mean_image) / std_image
# X_test = (X_test.astype(np.float32) - mean_image) / std_image


# In[4]:


X_train = X_train.reshape([X_train.shape[0],3,32,32]).transpose((0,2,3,1))
X_val = X_val.reshape([X_val.shape[0],3,32,32]).transpose((0,2,3,1))
X_test = X_test.reshape([X_test.shape[0],3,32,32]).transpose((0,2,3,1))

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape', X_test.shape)
print('Test labels shape', y_test.shape)


# # change ADAM?
# # Use better initializations for batch_norm?
# # add model saver/checkpointer. In addition keep track of best validation error and save that model

# In[19]:


inputs = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
random_rolls = tf.placeholder(dtype=tf.float32, shape=[54])
y_inputs = tf.placeholder(dtype=tf.uint8, shape=[None])
is_training = tf.placeholder(dtype=tf.bool)

lr = 0.1
weight_decay = 1e-4
momentum = 0.9
batch_size = 128
train_size = 45000
epochs = 500

out = architecture(inputs, random_rolls, is_training)
labels = tf.one_hot(y_inputs, 10)
softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=labels))
# confirm this is right
regularizer_loss = weight_decay * tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables('stoch_depth')
                                            if 'weights' in var.name])
loss = softmax_loss + regularizer_loss
training_summary = tf.summary.scalar('train_loss', loss)
step = tf.train.MomentumOptimizer(lr, momentum, use_nesterov=True).minimize(loss)
# step = tf.train.AdamOptimizer(lr).minimize(loss)
eve = evaluate(out, y_val)
validation_summary = tf.summary.scalar('Stoch_error', 100 * eve / len(y_val))


# In[ ]:


from datetime import datetime
now = datetime.now()
logdir = "tf_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"


# In[2]:


best_val = np.inf
with tf.Session() as sess: 
    
#     merge = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logdir, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    start_index = 0
    iter_number = -1
    for epoch in range(epochs):
        if epoch in (250, 375):
            lr /= 10
        for _ in range(train_size // batch_size):
            iter_number += 1
            if start_index + batch_size >= train_size:
                diff = start_index + batch_size - train_size
                indices = list(range(start_index, train_size)) + list(range(diff))
                start_index = diff
            else:
                indices = list(range(start_index, start_index + batch_size))
                start_index += batch_size

            X_train_batch, y_train_batch = X_train[indices], y_train[indices]
            # data augmentation
            x_offset = np.random.randint(-4, 4)
            y_offset = np.random.randint(-4, 4)
            flip_bool = np.random.uniform() > .5
            X_train_batch = np.roll(X_train_batch, (x_offset, y_offset), axis=(1, 2))
            if flip_bool:
                X_train_batch = np.flip(X_train_batch, axis=2)

            random_rolls_batch = np.random.uniform(size=54)
            _, loss_val, train_summ = sess.run([step, loss, training_summary], feed_dict={inputs: X_train_batch,
                                                                                          random_rolls: random_rolls_batch,
                                                                                          y_inputs: y_train_batch,
                                                                                          is_training: True})
            writer.add_summary(train_summ, iter_number)

            if iter_number % 100 == 0:
                val, val_summ = sess.run([eve, validation_summary], feed_dict={inputs: X_val,
                                                                               random_rolls: random_rolls_batch,
                                                                               is_training: False}) # random_rolls is irrelevant
                val = val * 100 / y_val.shape[0]
                    
                print('###########')
                print("epoch number {}, train loss {}, validation error {}".format(epoch + 1, loss_val, val))
                print("###########")
        
                # save the merge result summary
                writer.add_summary(val_summ, iter_number)
        if epoch % 10 == 0:
            save_path = saver.save(sess, 'checkpoints/model.ckpt')
            print("Model saved in file: %s" % save_path)
        if val < best_val:
            best_val = val
            save_path = saver.save(sess, 'checkpoints/best_validation.ckpt')
            print("Best validation model saved in file: %s" % save_path)
            
    writer.close()


# # Tests (ignore)

# In[ ]:


# # confirm tensorboard
# random_rolls_batch = np.random.uniform(size=54)
# X_train_batch = X_train[list(range(20))]
# with tf.Session() as sess:
#     writer = tf.summary.FileWriter('logs', sess.graph)
#     sess.run(tf.global_variables_initializer())
#     sess.run(out, feed_dict={inputs: X_train_batch,
#                              random_rolls: random_rolls_batch,
#                              is_training: True})
#     sess.run(out, feed_dict={inputs: X_val,
#                              random_rolls: random_rolls_batch,
#                              is_training: False})
#     writer.close()


# In[ ]:


# # confirm training = False gives same answers
# with tf.Session() as sess:
#     ls = []
#     with tf.variable_scope('stoch_depth', reuse=tf.AUTO_REUSE):
#         for i in range(3):
#             out = architecture(inputs, False)
#             if i == 0:
#                 sess.run(tf.global_variables_initializer())
#             ls.append(sess.run(out, feed_dict={inputs: X}))


# In[ ]:


# # confirm training = False but new init gives diff answers
# with tf.Session() as sess:
#     ls = []
#     with tf.variable_scope('stoch_depth', reuse=tf.AUTO_REUSE):
#         for i in range(3):
#             out = architecture(inputs, False)
#             sess.run(tf.global_variables_initializer())
#             ls.append(sess.run(out, feed_dict={inputs: X}))


# In[ ]:


# confirm training = True gives diff answers
# with tf.Session() as sess:
#     ls = []
#     with tf.variable_scope('stoch_depth', reuse=tf.AUTO_REUSE):
#         for i in range(3):
#             out = architecture(inputs, True)
#             if i == 0:
#                 sess.run(tf.global_variables_initializer())
#             ls.append(sess.run(out, feed_dict={inputs: X}))

