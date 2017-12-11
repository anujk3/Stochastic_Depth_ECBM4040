import tensorflow as tf

# W=(W−F+2P)/S+1

#####TODO############
# make train test global conditions instead of a switch in each layer.


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


def residual_block(inputs, output_size, survival_rate, random_roll,
                   is_training, scope, stride=1, padding='SAME'):
    # optional downsampling when strides > 1
    # optional stride param?
    # padding in case of different strides?

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        identity = inputs

        conv = tf.contrib.layers.conv2d(inputs,
                                        num_outputs=output_size,
                                        kernel_size=[3, 3],
                                        stride=stride,
                                        padding=padding,
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=tf.contrib.layers.batch_norm,
                                        normalizer_params={'updates_collections': None,
                                                           'is_training': is_training})

        # try changing initializer for gamma in batch_norm for this layer
        conv = tf.contrib.layers.conv2d(conv,
                                        num_outputs=output_size,
                                        kernel_size=[3, 3],
                                        stride=1,
                                        padding='SAME',
                                        activation_fn=None,
                                        normalizer_fn=tf.contrib.layers.batch_norm,
                                        normalizer_params={'scale': True,
                                                           'is_training': is_training,
                                                           'updates_collections': None})

        def training(conv):
            survives = tf.less(random_roll, survival_rate)
            conv = tf.cond(survives,
                           lambda: conv,
                           lambda: tf.zeros_like(conv),
                           name='survives_cond')
            return conv

        def testing(conv):
            return survival_rate * conv

        conv = tf.cond(tf.cast(is_training, tf.bool),
                       lambda: training(conv),
                       lambda: testing(conv),
                       name='is_trainig_cond')

        return tf.nn.relu(conv + identity)


def transition_block(inputs, output_size, survival_rate, random_roll,
                     is_training, scope, stride=1, padding='SAME'):

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

        conv = tf.contrib.layers.conv2d(inputs,
                                        num_outputs=output_size,
                                        kernel_size=[2, 2],
                                        stride=2,
                                        padding='VALID',
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=tf.contrib.layers.batch_norm,
                                        normalizer_params={'is_training': is_training,
                                                           'updates_collections': None})
        conv = tf.contrib.layers.conv2d(conv,
                                        num_outputs=output_size,
                                        kernel_size=[3, 3],
                                        stride=1,
                                        padding='SAME',
                                        activation_fn=None,
                                        normalizer_fn=tf.contrib.layers.batch_norm,
                                        normalizer_params={'scale': True,
                                                          'is_training': is_training,
                                                          'updates_collections': None})

        def training(conv):
            survives = tf.less(random_roll, survival_rate)
            return tf.cond(survives,
                           lambda: conv,
                           lambda: tf.zeros_like(conv),
                           name='survives_cond')

        def testing(conv):
            return survival_rate * conv

        conv = tf.cond(tf.cast(is_training, tf.bool),
                       lambda: training(conv),
                       lambda: testing(conv),
                       name='is_trainig_cond')

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
        tf.summary.scalar('Stoch_error', error_num)
    return error_num
