import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1), name="w1")
w2 = tf.Variable(tf.random_normal([2, 3], dtype=tf.float64, stddev=1), name="w2")

w1.assign(w2)

"""
程序将报错
 Input 'value' of 'Assign' Op has type float64 that does not match type float32 of argument 'ref'.
"""
