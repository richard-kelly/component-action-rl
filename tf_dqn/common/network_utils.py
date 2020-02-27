import tensorflow as tf
import numpy as np

# He weight initialization
weight_init = tf.variance_scaling_initializer(scale=2.0)


def get_activation(activation_name):
    if activation_name == 'relu':
        return tf.nn.relu
    if activation_name == 'leaky_relu':
        return tf.nn.leaky_relu
    else:
        raise ValueError(activation_name + ' is not a valid type of activation.')


def res_block(inputs, filters, kernel, is_training, activation='relu', downsample=False):
    act = get_activation(activation)
    stride = 2 if downsample else 1

    conv = tf.layers.conv2d(inputs, filters, kernel, strides=stride, padding='same', activation=None)
    conv = tf.layers.batch_normalization(conv, training=is_training)
    conv = act(conv)

    conv = tf.layers.conv2d(conv, filters, kernel, strides=1, padding='same', activation=None)
    conv = tf.layers.batch_normalization(conv, training=is_training)

    if conv.shape[1:] != inputs.shape[1:]:
        inputs = tf.layers.conv2d(inputs, filters, 1, strides=stride, padding='same', activation=None)

    return act(conv + inputs)


def res_block_preactivation(inputs, filters, kernel, is_training, activation='relu', downsample=False):
    act = get_activation(activation)
    stride = 2 if downsample else 1
    conv = tf.layers.batch_normalization(inputs, training=is_training)
    conv = act(conv)
    conv = tf.layers.conv2d(conv, filters, kernel, strides=stride, padding='same', activation=None)

    conv = tf.layers.batch_normalization(conv, training=is_training)
    conv = act(conv)
    conv = tf.layers.conv2d(conv, filters, kernel, strides=1, padding='same', activation=None)

    if conv.shape[1:] != inputs.shape[1:]:
        inputs = tf.layers.conv2d(inputs, filters, 1, strides=stride, padding='same', activation=None)

    return conv + inputs


def get_layers(inputs, spec, activation, is_training):
    # spec is a list, with various types as valid elements:
    #   'bn'                 - batch norm
    #   'relu'               - ReLU
    #   'leaky_relu'         - Leaky ReLU
    #   'flatten'            - reshapes to 2d, first dimension is batch size
    #   int                  - dense layer with 'int' units, then default activation, then bn
    #   []                   - recursive call to get_layers() for each in list, then concatenate all outputs on last dim
    #   {type: 'something'}  - various, depends on type (other parameters as indicated)
    #       max_pool         - pool_size, strides, padding (see tf.layers.max_pooling2d)
    #       avg_pool         - pool_size, strides, padding (see tf.layers.average_pooling2d)
    #       dense            - units (no activation or bn)
    #       resblock         - filters, kernel_size, count, downsample [optional], original [optional]
    #       conv_act_bn      - filters, kernel_size, stride [optional]
    #       conv             - filters, kernel_size, stride [optional]

    for part in spec:
        if type(part) is str:
            func = part.lower()
            if func == 'bn':
                inputs = tf.layers.batch_normalization(inputs, training=is_training)
            elif func == 'relu':
                act = get_activation('relu')
                inputs = act(inputs)
            elif func == 'leaky_relu':
                act = get_activation('leaky_relu')
                inputs = act(inputs)
            elif func == 'flatten':
                inputs = tf.reshape(inputs, shape=[-1, np.prod(inputs.shape[1:])])
            else:
                raise ValueError(part + ' is not a valid type of network part.')
        elif type(part) is int:
            inputs = tf.layers.dense(
                inputs,
                part,
                activation=get_activation(activation),
                kernel_initializer=weight_init
            )
            inputs = tf.layers.batch_normalization(inputs, training=is_training)
        elif type(part) is list:
            branches = []
            for sub_part in part:
                branches.append(get_layers(inputs, sub_part, activation, is_training))
            inputs = tf.concat(branches, axis=-1)
        elif type(part) is dict:
            func = part['type']
            if func == 'max_pool':
                inputs = tf.layers.max_pooling2d(
                    inputs=inputs,
                    pool_size=part['pool_size'],
                    strides=part['strides'],
                    padding=part['padding']
                )
            elif func == 'avg_pool':
                inputs = tf.layers.average_pooling2d(
                    inputs=inputs,
                    pool_size=part['pool_size'],
                    strides=part['strides'],
                    padding=part['padding']
                )
            elif func == 'dense':
                inputs = tf.layers.dense(
                    inputs,
                    part,
                    activation=None,
                    kernel_initializer=weight_init
                )
            elif func == 'resblock':
                downsample = False if 'downsample' not in part else part['downsample']
                for _ in range(part['count']):
                    if 'original' in part and part['original']:
                        inputs = res_block(inputs, part['filters'], part['kernel_size'], is_training, activation, downsample)
                    else:
                        inputs = res_block_preactivation(inputs, part['filters'], part['kernel_size'], is_training, activation, downsample)
            elif func == 'conv_act_bn':
                inputs = tf.layers.conv2d(
                    inputs=inputs,
                    filters=part['filters'],
                    kernel_size=part['kernel_size'],
                    strides=part['stride'] if 'stride' in part else 1,
                    padding='same',
                    activation=get_activation(activation),
                    kernel_initializer=weight_init
                )
                inputs = tf.layers.batch_normalization(inputs, training=is_training)
            elif func == 'conv':
                inputs = tf.layers.conv2d(
                    inputs=inputs,
                    filters=part['filters'],
                    kernel_size=part['kernel_size'],
                    strides=part['stride'] if 'stride' in part else 1,
                    padding='same',
                    activation=None,
                    kernel_initializer=weight_init
                )
            else:
                raise ValueError(func + ' is not a valid type of network part.')
        else:
            raise ValueError(type(part) + ' is not a valid type of network part.')

    return inputs


def get_conv_layers(inputs, spec, activation, is_training):
    # expecting spec to be a list of lists of dicts.
    # each inner list is a list of conv layers using the same input to be concatenated
    # each dict gives the number of filters and kernel size of a conv layer

    for conv_unit in spec:
        conv_layers = []
        for conv in conv_unit:
            conv_layer = tf.layers.conv2d(
                inputs=inputs,
                filters=conv['filters'],
                kernel_size=conv['kernel_size'],
                padding='same',
                activation=get_activation(activation),
                kernel_initializer=weight_init
            )
            conv_layer = tf.layers.batch_normalization(conv_layer, training=is_training)
            conv_layers.append(conv_layer)
        inputs = tf.concat(conv_layers, axis=-1)
    return inputs


def get_dense_layers(inputs, spec, activation, is_training):
    # expecting spec to be a list of ints
    for num_units in spec:
        dense = tf.layers.dense(
            inputs,
            units=num_units,
            activation=get_activation(activation),
            kernel_initializer=weight_init
        )
        dense = tf.layers.batch_normalization(dense, training=is_training)
        inputs = dense
    return inputs