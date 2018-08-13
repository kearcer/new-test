import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义placehoder作为存放输入数据的地方，这里的维度也不一定要定义。
# 但如果维度是确定的，那么给出维度数可以降低错误概率。
x = tf.placeholder(tf.float32, shape=(3, 2), name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

# 下面一行将报错：InvalidArgumentError (see above for traceback):
# You must feed a value for placeholder tensor 'input' with dtype float and shape [1,2]
# print(sess.run(y))

# 下面一行将会和3.4.2节中一样得到输出（3.95757794）
print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))
