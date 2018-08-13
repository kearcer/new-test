import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# 创建一个会话。
sess = tf.Session()
# 使用这个创建好的会话来得到关心的运算结果。比如可以调用sess.run(result)
# 来得到3.1节样例中的张量result的取值。
sess.run(...)
# 关闭会话使得本次运行中使用到的资源可以被释放
sess.close()