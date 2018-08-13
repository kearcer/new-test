import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# 创建一个会话，并通过Python中的上下文管理器来管理这个会话
with tf.Session() as sess:
    # 使用创建好的会话来计算关心的结果。
    sess.run(...)
# 不需要再调用“Session.close（）”函数来关闭对话框
# 当上下文退出时对话关闭资源解释器也自动释放完成。