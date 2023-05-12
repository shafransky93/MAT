import numpy as np
import tensorflow as tf
import math

def positional_encoding(maximum_position_encoding, d_model):
    # create a 2D grid of positions
    position = tf.range(maximum_position_encoding, dtype=tf.float32)[:, tf.newaxis]
    # compute the angle frequencies for each position
    div_term = tf.math.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(math.log(10000.0) / d_model))
    # compute the positional encodings
    sin = tf.math.sin(position * div_term)
    cos = tf.math.cos(position * div_term)
    # concatenate the sin and cos arrays
    pos_encoding = tf.concat([sin, cos], axis=-1)
    # reshape to match the expected shape of (1, max_length, d_model)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

max_length = 6000
d_model = 512
pos_encoding = positional_encoding(max_length, d_model)
