import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# 计算张量的取值有以下两种方法
sess = tf.Session()
with sess.as_default():
    print(result.eval())

sess = tf.Session()

# 以下两个命令有相同的功能
print(sess.run(result))
print(result.eval(session=sess))