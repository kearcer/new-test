import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# 改变需要通过设置参数validate_shape=False

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1), name="w1")
w2 = tf.Variable(tf.random_normal([2, 2], stddev=1), name="w2")
# 下面这句话会报维度不匹配的错误：
# VariableError:Dimension 1 in both shapes must be equal, but are 3 and 2. Shapes are [2,3] and [2,2]. for 'Assign'
# (op: 'Assign') with input shapes: [2,3], [2,2].
#
# tf.assign(w1, w2)

# 这句却被成功执行
tf.assign(w1, w2, validate_shape=False)
