import tensorflow as tf

a = tf.constant([-1.0, 2.0])

# 激励函数
with tf.Session() as sess:
    b = tf.nn.relu(a)
    print(sess.run(b))
    
    c = tf.sigmoid(a)
    print(sess.run(c))
