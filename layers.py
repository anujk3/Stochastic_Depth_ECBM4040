import tensorflow as tf

# W=(W−F+2P)/S+1


def first_layer(inputs, is_training, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        out = tf.contrib.layers.conv2d(inputs,
                                       num_outputs=16,
                                       kernel_size=(3, 3),
                                       stride=1,
                                       padding='SAME',
                                       activation_fn=tf.nn.relu,
                                       normalizer_fn=tf.contrib.layers.batch_norm,
                                       normalizer_params={'updates_collections': None,
                                                          'is_training': is_training})
    return out


def conv_block(inputs, output_size, first_kernel_size, first_stride,
               first_padding, is_training):
    conv = tf.contrib.layers.conv2d(inputs,
                                    num_outputs=output_size,
                                    kernel_size=first_kernel_size,
                                    stride=first_stride,
                                    padding=first_padding,
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                    normalizer_params={'updates_collections': None,
                                                       'is_training': is_training})

    # try changing initializer for gamma in batch_norm for this layer
    conv = tf.contrib.layers.conv2d(conv,
                                    num_outputs=output_size,
                                    kernel_size=(3, 3),
                                    stride=1,
                                    padding="SAME",
                                    activation_fn=None,
                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                    normalizer_params={'scale': True,
                                                       'is_training': is_training,
                                                       'updates_collections': None})
    return conv


def residual_block(inputs, output_size, survival_rate, random_roll,
                   is_training, scope):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        identity = inputs

        survives = tf.less(random_roll, survival_rate)

        conv = tf.cond(tf.logical_and(is_training, tf.logical_not(survives)),
                       lambda: tf.zeros_like(identity),
                       lambda: conv_block(inputs, output_size, (3, 3), 1, "SAME",
                                          is_training),
                       name='first_cond')
        conv = tf.cond(is_training,
                       lambda: conv,
                       lambda: survival_rate * conv,
                       name='second_cond')

        return tf.nn.relu(conv + identity)


def transition_block(inputs, output_size, survival_rate, random_roll,
                     is_training, scope):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        identity = tf.contrib.layers.avg_pool2d(inputs,
                                                kernel_size=(2, 2),
                                                stride=2,
                                                padding='VALID')
        # confirm. Anuj may be right. seems like they did zero padding instead of this.
        identity = tf.contrib.layers.conv2d(identity,
                                            num_outputs=output_size,
                                            kernel_size=[1, 1],
                                            stride=1,
                                            padding='VALID')

        survives = tf.less(random_roll, survival_rate)

        conv = tf.cond(tf.logical_and(is_training, tf.logical_not(survives)),
                       lambda: tf.zeros_like(identity),
                       lambda: conv_block(inputs, output_size, (2, 2), 2, "VALID",
                                          is_training),
                       name='first_cond')

        conv = tf.cond(is_training,
                       lambda: conv,
                       lambda: survival_rate * conv,
                       name='second_cond')

        return tf.nn.relu(conv + identity)


# output layer could be the issue?
def output_layer(inputs, scope, output_size=10):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        out = tf.contrib.layers.avg_pool2d(inputs,
                                           kernel_size=inputs.shape[1:3],
                                           stride=1,
                                           padding='VALID')
        out = tf.contrib.layers.flatten(out)
        out = tf.contrib.layers.fully_connected(out,
                                                output_size,
                                                activation_fn=None)
    return out


def architecture(inputs, random_rolls, is_training, P=0.5, L=54):
    with tf.variable_scope('stoch_depth', reuse=tf.AUTO_REUSE):

        out = first_layer(inputs, is_training, 'input')

        l = 1
        with tf.variable_scope('stack1', reuse=tf.AUTO_REUSE):
            for i in range(1, 19):
                p = 1 - l / L * (1 - P)
                out = residual_block(out, 16, p, random_rolls[l-1], is_training, 'res'+str(i))
                l += 1

        with tf.variable_scope('stack2', reuse=tf.AUTO_REUSE):
            p = 1 - l / L * (1 - P)
            out = transition_block(out, 32, p, random_rolls[l-1], is_training, 'res'+str(1))
            l += 1

            for i in range(2, 19):
                p = 1 - l / L * (1 - P)
                out = residual_block(out, 32, p, random_rolls[l-1], is_training, 'res'+str(i))
                l += 1

        with tf.variable_scope('stack3', reuse=tf.AUTO_REUSE):
            p = 1 - l / L * (1 - P)
            out = transition_block(out, 64, p, random_rolls[l-1], is_training, 'res'+str(1))
            l += 1

            for i in range(2, 19):
                p = 1 - l/L * (1 - P)
                out = residual_block(out, 64, p, random_rolls[l-1], is_training, 'res'+str(i))

        out = output_layer(out, 'out', 10)

        return out


def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
    return error_num
