import tensorflow as tf

# confirm batch norm betweren activation and input (or vice-versa?)
# scale? the link below says may not be required if relu is next layer.
# NOTE from https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm?
# implement variable probs in notebook itself.
# W=(W−F+2P)/S+1

#####TODO############
# make train test global conditions instead of a switch in each layer.

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
                   random_roll,
                   is_training, scope, stride=1, padding='SAME'):
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
                                                           'is_training': is_training})
        out = tf.contrib.layers.conv2d(conv,
                                       num_outputs=output_size,
                                       kernel_size=[3, 3],
                                       stride=1,
                                       padding='SAME',
                                       activation_fn=None,
                                       normalizer_fn=tf.contrib.layers.batch_norm,
                                       normalizer_params={'scale': True,
                                                          'is_training': is_training})

        def training(out, identity):
            survives = tf.less(random_roll, survival_rate)
            return tf.cond(survives,
                           lambda: out + identity,
                           lambda: identity,
                           name='survives_cond')

        def testing(out, identity):
            return survival_rate * out + identity

        out = tf.cond(is_training,
                      lambda: training(out, identity),
                      lambda: testing(out, identity),
                      name='is_trainig_cond')

        return tf.nn.relu(out)

def transition_block(inputs, output_size, survival_rate,
                     random_roll,
                     is_training, scope, stride=1, padding='SAME'):

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
                                                           'is_training': is_training})
        out = tf.contrib.layers.conv2d(conv,
                                       num_outputs=output_size,
                                       kernel_size=[3, 3],
                                       stride=1,
                                       padding='SAME',
                                       activation_fn=None,
                                       normalizer_fn=tf.contrib.layers.batch_norm,
                                       normalizer_params={'scale': True,
                                                          'is_training': is_training})
        
        def training(out, identity):
            survives = tf.less(random_roll, survival_rate)
            return tf.cond(survives,
                           lambda: out + identity,
                           lambda: identity,
                           name='survives_cond')

        def testing(out, identity):
            return survival_rate * out + identity

        out = tf.cond(is_training,
                      lambda: training(out, identity),
                      lambda: testing(out, identity),
                      name='is_trainig_cond')

        return tf.nn.relu(out)


def output_layer(inputs, scope, output_size=10):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        pooling = tf.contrib.layers.avg_pool2d(inputs,
                                               kernel_size=inputs.shape[1:3],
                                               stride=1,
                                               padding='VALID')
        flatten = tf.contrib.layers.flatten(pooling)
        fc = tf.contrib.layers.fully_connected(flatten,
                                               output_size)
    return fc


def architecture(inputs, random_rolls, is_training, P=0.5, L=54):
    out = first_layer(inputs, is_training, 'input')
    l = 1

    with tf.variable_scope('stack1', reuse=tf.AUTO_REUSE):
        for i in range(1, 19):
            p = 1 - l/L * (1 - P)
            out = residual_block(out, 16, p, random_rolls[l-1], is_training, 'res'+str(i))
            l += 1

    with tf.variable_scope('stack2', reuse=tf.AUTO_REUSE):
        p = 1 - l/L * (1 - P)
        out = transition_block(out, 32, p, random_rolls[l-1], is_training, 'res'+str(1))
        l += 1

        for i in range(2, 19):
            p = 1 - l/L * (1 - P)
            out = residual_block(out, 32, p, random_rolls[l-1], is_training, 'res'+str(i))
            l += 1

    with tf.variable_scope('stack3', reuse=tf.AUTO_REUSE):
        p = 1 - l/L * (1 - P)
        out = transition_block(out, 64, p, random_rolls[l-1], is_training, 'res'+str(1))
        l += 1

        for i in range(2, 19):
            p = 1 - l/L * (1 - P)
            out = residual_block(out, 64, p, random_rolls[l-1], is_training, 'res'+str(i))

    return output_layer(out, 'out', 10)

def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('Stoch_error', error_num)
    return error_num