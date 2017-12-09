import tensorflow as tf
import numpy as np

# confirm batch norm betweren activation and input (or vice-versa?)
# scale? the link below says may not be required if relu is next layer.
# NOTE from https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm?
# implement variable probs in notebook itself.
# W=(W−F+2P)/S+1
def first_layer(inputs, training, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        out = tf.contrib.layers.conv2d(inputs,
                                       num_outputs=16,
                                       kernel_size=(3, 3),
                                       stride=1,
                                       padding='SAME',
                                       activation_fn=tf.nn.relu,
                                       normalizer_fn=tf.contrib.layers.batch_norm,
                                       normalizer_params={'scale': True,
                                                          'is_training': training})
    return out

def residual_block(inputs, output_size, survival_rate,
                   training, scope, stride=1, padding='SAME'):
    #optional downsampling when strides > 1
    # optional stride param?
    #padding in case of different strides?

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        identity = inputs

        conv = tf.contrib.layers.conv2d(inputs,
                                        num_outputs=output_size,
                                        kernel_size=[3, 3],
                                        stride=stride,
                                        padding=padding,
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=tf.contrib.layers.batch_norm,
                                        normalizer_params={'scale': True,
                                                           'is_training': True})
        out = tf.contrib.layers.conv2d(conv,
                                       num_outputs=output_size,
                                       kernel_size=[3, 3],
                                       stride=1,
                                       padding='SAME',
                                       activation_fn=None,
                                       normalizer_fn=tf.contrib.layers.batch_norm,
                                       normalizer_params={'scale': True,
                                                          'is_training': True})
        if training:
            bernoulli = tf.random_uniform(shape=[], minval=0.0, maxval=1.0)
            survives = tf.less(bernoulli, survival_rate)
            out = tf.cond(survives,
                          lambda: out + identity,
                          lambda: identity)
            return tf.nn.relu(out)
        else:
            out *= survival_rate
            return tf.nn.relu(out + identity)

def transition_block(inputs, output_size, survival_rate,
                     training, scope, stride=1, padding='SAME'):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        avg_pool = tf.contrib.layers.avg_pool2d(inputs,
                                                kernel_size=(2, 2),
                                                stride=2,
                                                padding='VALID')
        # confirm. Anuj may be right. seems like they did zero padding instead of this.
        identity = tf.contrib.layers.conv2d(avg_pool,
                                            num_outputs=output_size,
                                            kernel_size=[1, 1],
                                            stride=1,
                                            padding='VALID')
        # confirm. which layer is used for downsampling?
        conv = tf.contrib.layers.conv2d(inputs,
                                        num_outputs=output_size,
                                        kernel_size=[2, 2],
                                        stride=2,
                                        padding='VALID',
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=tf.contrib.layers.batch_norm,
                                        normalizer_params={'scale': True,
                                                           'is_training': True})
        out = tf.contrib.layers.conv2d(conv,
                                       num_outputs=output_size,
                                       kernel_size=[3, 3],
                                       stride=1,
                                       padding='SAME',
                                       activation_fn=None,
                                       normalizer_fn=tf.contrib.layers.batch_norm,
                                       normalizer_params={'scale': True,
                                                          'is_training': True})
        if training:
            bernoulli = tf.random_uniform(shape=[], minval=0.0, maxval=1.0)
            survives = tf.less(bernoulli, survival_rate)
            out = tf.cond(survives,
                          lambda: out + identity,
                          lambda: identity)
            return tf.nn.relu(out)
        else:
            out *= survival_rate
            return tf.nn.relu(out + identity)


def output_layer(inputs, scope, output_size=10):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        pooling = tf.contrib.layers.avg_pool2d(inputs,
                                               kernel_size=inputs.shape[1:3],
                                               stride=1,
                                               padding='VALID')
        flatten = tf.contrib.layers.flatten(pooling)
        fc = tf.contrib.layers.fully_connected(flatten,
                                               output_size,
                                               activation_fn=tf.nn.softmax)
    return fc
