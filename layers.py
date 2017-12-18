import tensorflow as tf


def first_layer(inputs, is_training, num_outputs=16, scope='Inputs'):
    '''
    conv2d -> batch_norm -> relu

    Parameters
    ----------
    inputs: input tensor with shape "NHWC"
    is_training: boolean tensor to indicate training/testing mode
    num_outputs: (default: 16) output depth
    scope: (default: "Inputs") variable scope

    Output
    ------
    tensor of shape "NHWC" where C=num_outputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        out = tf.contrib.layers.conv2d(inputs,
                                       num_outputs=num_outputs,
                                       kernel_size=(3, 3),
                                       stride=1,
                                       padding='SAME',
                                       activation_fn=tf.nn.relu,
                                       normalizer_fn=tf.contrib.layers.batch_norm,
                                       normalizer_params={'updates_collections': None,
                                                          'is_training': is_training,
                                                          'scale': True})
    return out


def conv_block(inputs, output_size, first_kernel_size, first_stride,
               first_padding, is_training):
    '''
    conv2d -> batch_norm -> relu -> conv2d -> batch_norm

    Creates block with two back to back convolution layers.
    This is the building block for a single residual block.

    Parameters
    ----------
    inputs: input tensor with shape "NHWC"
    output_size: number of output channels
    first_kernel_size: usually 3x3, except in transition block, where it is 2x2
    first_stride: usually 1, except in transition block, where it is 2
    first_padding: usually "SAME", except in transition block, where it is "VALID"
    is_training: tensorflow boolean indicating wheter training or testing mode
    '''
    conv = tf.contrib.layers.conv2d(inputs,
                                    num_outputs=output_size,
                                    kernel_size=first_kernel_size,
                                    stride=first_stride,
                                    padding=first_padding,
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                    normalizer_params={'updates_collections': None,
                                                       'is_training': is_training,
                                                       'scale': True})

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
    '''
    -> conv_block -> + -> relu
      |              ^
      |_ _ _ _ _ _ _ |

    Basic residual block with stochastic depth. During testing, 
    the layers are not dropped. Instead, the output is multiplied
    by the survival rate.

    Parameters
    ----------
    inputs: input tensor of shape "NHWC"
    output_size: number of output channels (same as input channels)
    survival_rate: scalar defining the survival rate of the layer
    random_roll: random number between [0, 1) generated every epoch
    is_training: boolean specifying training/testing mode
    scope: string to specify variable scope

    Output
    ------
    tensor with shape "NHWC"
    '''

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        identity = inputs

        survives = tf.less(random_roll, survival_rate)

        conv = tf.cond(tf.logical_and(is_training, tf.logical_not(survives)),
                       lambda: tf.zeros_like(identity),
                       lambda: conv_block(inputs, output_size,
                                          first_kernel_size=(3, 3),
                                          first_stride=1, first_padding="SAME",
                                          is_training=is_training),
                       name='first_cond')
        conv = tf.cond(is_training,
                       lambda: conv,
                       lambda: survival_rate * conv,
                       name='second_cond')

        return tf.nn.relu(conv + identity)


def transition_block(inputs, output_size, survival_rate, random_roll,
                     is_training, scope, strategy='pad'):
    '''
    -> conv_block - - - - > + -> relu
      |                     ^
      |                     |
       -> avg_pool -> pad ->

    Modified residual block. This is the first cell of the second
    and third stacks. This halves the image width and height, but
    doubles the number of channels, thus maintaining time complexity.
    During testing, the layer is not dropped. Instead, the output is
    multiplies by the survival rate

    Parameters
    ----------
    inputs: input tensor of shape "NHWC"
    output_size: number of output channels (double of input channels)
    survival_rate: scalar defining the survival rate of the layer
    random_roll: random number between [0, 1) generated every epoch
    is_training: boolean specifying training/testing mode
    scope: string to specify variable scope
    strategy: 'pad' or 'conv'. method to follow to increase the number of
              channels of identity connection. Iriginal paper uses 'pad'

    Output
    ------
    tensor with shape "NHWC" where H and W are half of original imahge
    and C is double
    '''

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        identity = tf.contrib.layers.avg_pool2d(inputs,
                                                kernel_size=(2, 2),
                                                stride=2,
                                                padding='VALID')

        if strategy == 'conv':
            identity = tf.contrib.layers.conv2d(identity,
                                                num_outputs=output_size,
                                                kernel_size=[1, 1],
                                                stride=1,
                                                padding='VALID')
        elif strategy == 'pad':
            padding_size = (output_size - int(inputs.shape[-1])) // 2
            paddings = tf.constant([[0, 0],
                                    [0, 0],
                                    [0, 0],
                                    [padding_size, padding_size]])
            identity = tf.pad(identity,
                              paddings)

        survives = tf.less(random_roll, survival_rate)

        conv = tf.cond(tf.logical_and(is_training, tf.logical_not(survives)),
                       lambda: tf.zeros_like(identity),
                       lambda: conv_block(inputs, output_size,
                                          first_kernel_size=(2, 2),
                                          first_stride=2,
                                          first_padding="VALID",
                                          is_training=is_training),
                       name='first_cond')

        conv = tf.cond(is_training,
                       lambda: conv,
                       lambda: survival_rate * conv,
                       name='second_cond')

        return tf.nn.relu(conv + identity)


def output_layer(inputs, scope, output_size=10):
    '''
    -> avg_pool -> flatten -> fully_connected
    Parameters
    ----------
    inputs: input tensor of shape "NHWC"
    scope: string to define variable scope
    output_size: 10 for CIFAR-10
    '''
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


def architecture(inputs, random_rolls, is_training, strategy, P=0.5, L=54,
                 scope='stoch_depth'):
    '''
    Builds the stochastic depth network. Current implementation consists of 3 stacks
    of 18 residual blocks each. Thus, counting the 2 convolution layers per residual
    block, the input layer and the output layer, the total number of layers comes
    out to be 110.

    Inputs
    -----
    inputs: input tensor of shape NHWC
    random_rolls: a tensor of shape (L,) where each value is a number from [0, 1),
                  randomly generated for each mini-batch
    is_training: boolean to indicate training/testing mode
    strategy: 'pad' or 'conv'. Strategy to match dimensions of the identity function
              in the transition block. The paper uses 'pad'
    P: The probability of survival of the input layer
    L: Total residual blocks in the network. Needs to be a multiple of 3 so that all 
       stacks can have an equal number of residual blocks.
    '''
    num_stacks = 3
    assert L % num_stacks == 0
    res_per_stack = L // num_stacks

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        out = first_layer(inputs, is_training)

        survival_rates = [1 - l / L * (1 - P) for l in range(1, L+1)]

        l = 0
        with tf.variable_scope('stack1', reuse=tf.AUTO_REUSE):
            for i in range(res_per_stack):
                out = residual_block(inputs=out, output_size=16, survival_rate=survival_rates[l],
                                     random_roll=random_rolls[l], is_training=is_training,
                                     scope='res'+str(i))
                l += 1

        with tf.variable_scope('stack2', reuse=tf.AUTO_REUSE):
            out = transition_block(inputs=out, output_size=32, survival_rate=survival_rates[l],
                                   random_roll=random_rolls[l], is_training=is_training,
                                   scope='res'+str(0), strategy=strategy)
            l += 1

            for i in range(1, res_per_stack):
                out = residual_block(inputs=out, output_size=32, survival_rate=survival_rates[l],
                                     random_roll=random_rolls[l], is_training=is_training,
                                     scope='res'+str(i))
                l += 1

        with tf.variable_scope('stack3', reuse=tf.AUTO_REUSE):
            out = transition_block(inputs=out, output_size=64, survival_rate=survival_rates[l],
                                   random_roll=random_rolls[l], is_training=is_training,
                                   scope='res'+str(0), strategy=strategy)
            l += 1

            for i in range(1, res_per_stack):
                out = residual_block(inputs=out, output_size=64, survival_rate=survival_rates[l],
                                     random_roll=random_rolls[l], is_training=is_training,
                                     scope='res'+str(i))
                l += 1

        out = output_layer(out, scope='out', output_size=10)

        return out


def loss_function(logits, labels, weight_decay, arch_scope):
    '''
    Calculates softmax_loss and adds l2 regularizer on the weights

    Parameters
    ----------
    logits: tensor representing unnormalized decision function for each class
    labels: one hot encoded true labels
    weight_decay: scalar representing regularization constant
    arch_scope: scope name of the full architecture. Required to determine all
                the trainable parameters in the architecture
    '''
    softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=labels))
    regularizer_loss = weight_decay * tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables(arch_scope)
                                                if 'weights' in var.name])
    return softmax_loss + regularizer_loss
