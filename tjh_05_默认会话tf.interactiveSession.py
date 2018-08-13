import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

sess = tf.InteractiveSession
print(result.eval())
sess.close()