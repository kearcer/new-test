import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# 使用sigmoid函数将y转换为0-1之间的数值。转换后y代表预测是正样本数据的概率，1-y代表的是预测负样本的概率
y = tf.sigmoid(y)
# 定义损失函数来刻画预测值与真实值之间的差距
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    +(1-y)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0))
)

# 定义学习率，再第4章中将更加具体介绍学习率
learning_rate = 0.001
# 定义反响传播算法来优化神经网络中的参数。
train_step =\
    tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
