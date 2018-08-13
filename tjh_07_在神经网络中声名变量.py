import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
weights = tf.Variable(tf.random_normal([2, 3], stddev=2))
print(weights)

# 生成初始值为0的随机数
biases = tf.Variable(tf.zero([3]))

w2 = tf.Variable(weights.initial_value())
w3 = tf.Variable(weights.initial_value() * 2.0)