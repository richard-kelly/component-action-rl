import tensorflow as tf


def get_conv_layers(inputs, spec, use_bn, is_training, propagate_inputs):
    # expecting spec to be a list of lists of dicts.
    # each inner list is a list of conv layers using the same input to be concatenated
    # each dict gives the number of filters and kernel size of a conv layer
    original_layers = inputs

    for conv_unit in spec:
        conv_layers = []
        for conv in conv_unit:
            conv_layer = tf.layers.conv2d(
                inputs=inputs,
                filters=conv['filters'],
                kernel_size=conv['kernel_size'],
                padding='same',
                activation=tf.nn.leaky_relu
            )
            if use_bn:
                conv_layer = tf.layers.batch_normalization(conv_layer, training=is_training)
            conv_layers.append(conv_layer)
        if propagate_inputs:
            conv_layers.append(original_layers)
        inputs = tf.concat(conv_layers, axis=-1)
    return inputs


def get_dense_layers(inputs, spec, use_bn, is_training):
    # expecting spec to be a list of ints
    for num_units in spec:
        dense = tf.layers.dense(
            inputs,
            units=num_units,
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.variance_scaling_initializer(scale=2.0)
        )
        if use_bn:
            dense = tf.layers.batch_normalization(dense, training=is_training)
        inputs = dense
    return inputs
