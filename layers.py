import tensorflow as tf
import numpy as np

# confirm batch norm betweren activation and input (or vice-versa?)
# scale? the link below says may not be required if relu is next layer.
# NOTE from https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm?
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
                   training, outer_scope, scope, strides=[1,1], padding='SAME'):
    #optional downsampling when strides > 1
    # optional stride param?
    #padding in case of different strides?
    with tf.name_scope(outer_scope):
        with tf.name_scope(scope):

            identity = inputs

            bernoulli = np.random.uniform()
            survives = bernoulli < survival_rate

            if training:
                if survives:
                    conv = tf.contrib.layers.conv2d(inputs,
                                                    num_outputs=output_size,
                                                    kernel_size=[3, 3],
                                                    strides=strides,
                                                    padding=padding,
                                                    activation_fn=tf.nn.relu,
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params={'scale': True,
                                                                       'is_training': True})
                    out = tf.contrib.layers.conv2d(conv,
                                                   num_outputs=output_size,
                                                   kernel_size=[3, 3],
                                                   strides=[1,1],
                                                   padding='SAME',
                                                   activation_fn=None,
                                                   normalizer_fn=tf.contrib.layers.batch_norm,
                                                   normalizer_params={'scale': True,
                                                                      'is_training': True})
                else:
                    out = tf.zeros(shape=inputs.shape)
            else:
                conv = tf.contrib.layers.conv2d(inputs,
                                                num_outputs=output_size,
                                                kernel_size=[3, 3],
                                                strides=strides,
                                                padding=padding,
                                                activation_fn=tf.nn.relu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,
                                                normalizer_params={'scale': True,
                                                                   'is_training': False})
                out = tf.contrib.layers.conv2d(conv,
                                               num_outputs=output_size,
                                               kernel_size=[3, 3],
                                               strides=[1,1],
                                               padding='SAME',
                                               activation_fn=None,
                                               normalizer_fn=tf.contrib.layers.batch_norm,
                                               normalizer_params={'scale': True,
                                                                  'is_training': False})
                out *= survival_rate


    return tf.contrib.layers.relu(out + identity)

# def transition_block(inputs, output_size, survival_rate,
#                      training, outer_scope, scope, strides=[1,1], padding='SAME'):

#     with tf.name_scope(outer_scope):
#         with tf.name_scope(scope):
#             avg_pool = tf.contrib.layers.avg_pool2d(input,
#                                                     kernel_size=(2, 2),
#                                                     stride=2,
#                                                     padding='VALID')
#             identity = tf.contrib.layers.conv2d(avg_pool,
#                                                 num_outputs=output_size,
#                                                 kernel_size=[1, 1],
#                                                 stride=1,
#                                                 padding='SAME')

#             bernoulli = np.random.uniform()
#             survives = bernoulli < survival_rate

#             if training:
#                 if survives:
#                     conv = tf.contrib.layers.conv2d(inputs,
#                                                     num_outputs=output_size,
#                                                     kernel_size=[3, 3],
#                                                     strides=strides,
#                                                     padding=padding,
#                                                     activation_fn=tf.nn.relu,
#                                                     normalizer_fn=tf.contrib.layers.batch_norm,
#                                                     normalizer_params={'scale': True,
#                                                                        'is_training': True})
#                     out = tf.contrib.layers.conv2d(conv,
#                                                    num_outputs=output_size,
#                                                    kernel_size=[3, 3],
#                                                    strides=[1,1],
#                                                    padding='SAME',
#                                                    activation_fn=None,
#                                                    normalizer_fn=tf.contrib.layers.batch_norm,
#                                                    normalizer_params={'scale': True,
#                                                                       'is_training': True})
#                 else:
#                     out = tf.constant(0, shape=inputs.shape)

#             else:
#                 conv = tf.contrib.layers.conv2d(inputs,
#                                                 num_outputs=output_size,
#                                                 kernel_size=[3, 3],
#                                                 strides=strides,
#                                                 padding=padding,
#                                                 activation_fn=tf.nn.relu,
#                                                 normalizer_fn=tf.contrib.layers.batch_norm,
#                                                 normalizer_params={'scale': True,
#                                                                    'is_training': False})
#                 out = tf.contrib.layers.conv2d(conv,
#                                                num_outputs=output_size,
#                                                kernel_size=[3, 3],
#                                                strides=[1,1],
#                                                padding='SAME',
#                                                activation_fn=None,
#                                                normalizer_fn=tf.contrib.layers.batch_norm,
#                                                normalizer_params={'scale': True,
#                                                                   'is_training': False})
#                 out *= survival_rate

#     return tf.contrib.layers.relu(out + identity)


# def output_layer(input, scope, output_size=10):
#     with tf.name_scope(scope):
#         flatten = tf.contrib.layers.flatten(input)
#         fc = tf.contrib.layers.fully_connected(flatten,
#                                                output_size,
#                                                activation_fn=tf.nn.softmax)
#     return fc