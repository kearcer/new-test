import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# 声明w1，w2两个变量，这里还通过seed参数设定了随机种子
# 这样可以保证每次运行得到的结果是一致的
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

# 暂时将输入的特定向量定义为一个常值。注意这里的x是一个1x2的矩阵
x = tf.constant([[0.7, 0.9]])

# 通过3.4.2节描述的前向算法来获得神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
# 与3.4.2中计算的不同，这里不能直接通过sess.run(y)来获取y的值
# 因为w1，w2都没有运行初始化过程，以下两行分别初始化了w1，w2两个变量
# sess.run(w1.initializer)  # 初始化 w1
# sess.run(w2.initializer)  # 初始化 w2

# 初始化方法2
init_op = tf.global_variables_initializer()
sess.run(init_op)

# 输出[[3.95757794]].
print(sess.run(y))
sess.close()