import tensorflow as tf

def first_layer():
    pass

def residual_block(input, output_size, kernel_size,
                   strides, padding, survived, mode, transition,
                   outer_scope, scope):
    with tf.name_scope(out_scope):
        with tf.name_scope(scope):
            if survived:
                conv = tf.contrib.layers.conv2d(input, output_size, kernel_size,
                                                strides, padding, activation_fn=tf.nn.relu,
                                                normalizer_fn=tf.nn.batch_normalization,
                                                normalizer_params=blahblah, blahblah)
                out = tf.contrib.layers.conv2d(conv, output_size, kernel_size,
                                               strides, padding, activation_fn=None,
                                               normalizer_fn=tf.nn.batch_normalization,
                                               blahblah)
            else:
                out = tf.constant(0, shape=(, output_size))

            if transition:
                avg_pool = tf.contrib.layers.avg_pool2d(input, kernel_size=(2, 2), stride=2,
                                                        padding='VALID')
                identity = tf.contrib.layers.conv2d(avg_pool, output_size, kernel_size)
            else:
                identity = input

    return tf.contrib.layers.relu(out + identity)

def output_layer():
    pass


def blocks(input, output_shape, num_layers, outer_scope):
    for i in range(num_layers - 1):
        pass



