import tensorflow as tf

# Nothing happening here
x = tf.constant([[1, 2]])
print(x)
neg_x = tf.negative(x)
print(neg_x)

# sess.run actually computes values
with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(neg_x))
    