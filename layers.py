import tensorflow as tf
import numpy as np

# confirm batch norm betweren activation and input (or vice-versa?)
# scale? the link below says may not be required if relu is next layer.
# NOTE from https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm?
# implement variable probs in notebook itself.
def first_layer(inputs, training, scope):
    with tf.name_scope(scope):
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
                   training, outer_scope, scope, stride=1, padding='SAME'):
    #optional downsampling when strides > 1
    # optional stride param?
    #padding in case of different strides?
    bernoulli = np.random.uniform()
    survives = bernoulli < survival_rate

    with tf.name_scope(outer_scope):
        with tf.name_scope(scope):

            identity = inputs

            if training:
                if survives:
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
                    return tf.nn.relu(out + identity)
                else:
                    return tf.nn.relu(identity)
            else:
                conv = tf.contrib.layers.conv2d(inputs,
                                                num_outputs=output_size,
                                                kernel_size=[3, 3],
                                                stride=stride,
                                                padding=padding,
                                                activation_fn=tf.nn.relu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,
                                                normalizer_params={'scale': True,
                                                                   'is_training': False})
                out = tf.contrib.layers.conv2d(conv,
                                               num_outputs=output_size,
                                               kernel_size=[3, 3],
                                               stride=1,
                                               padding='SAME',
                                               activation_fn=None,
                                               normalizer_fn=tf.contrib.layers.batch_norm,
                                               normalizer_params={'scale': True,
                                                                  'is_training': False})
                out *= survival_rate

                return tf.nn.relu(out + identity)

def transition_block(inputs, output_size, survival_rate,
                     training, outer_scope, scope, stride=1, padding='SAME'):

    bernoulli = np.random.uniform()
    survives = bernoulli < survival_rate

    with tf.name_scope(outer_scope):
        with tf.name_scope(scope):
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
            if training:
                if survives:
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
                    return tf.nn.relu(out + identity)
                else:
                    return tf.nn.relu(identity)

            else:
                conv = tf.contrib.layers.conv2d(inputs,
                                                num_outputs=output_size,
                                                kernel_size=[2, 2],
                                                stride=2,
                                                padding='VALID',
                                                activation_fn=tf.nn.relu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,
                                                normalizer_params={'scale': True,
                                                                   'is_training': False})
                out = tf.contrib.layers.conv2d(conv,
                                               num_outputs=output_size,
                                               kernel_size=[3, 3],
                                               stride=1,
                                               padding='SAME',
                                               activation_fn=None,
                                               normalizer_fn=tf.contrib.layers.batch_norm,
                                               normalizer_params={'scale': True,
                                                                  'is_training': False})
                out *= survival_rate

                return tf.nn.relu(out + identity)


# def output_layer(input, scope, output_size=10):
#     with tf.name_scope(scope):
#         flatten = tf.contrib.layers.flatten(input)
#         fc = tf.contrib.layers.fully_connected(flatten,
#                                                output_size,
#                                                activation_fn=tf.nn.softmax)
#     return fc