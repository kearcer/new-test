import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# tf.constant 是一个计算，这个计算结果为一个张量，并且保存在变量a中
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="add")
result = tf.add(a, b, name="add")
print (result)
"""
输出：Tensor("add:0", shape=(2,), dtype=float32)
"""
