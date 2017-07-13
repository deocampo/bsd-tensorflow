


import tensorflow as tf

# Basic constant operations
# The value returned by the constructor represents the output
# of the Constant op.
c1 = tf.constant(4)
c2 = tf.constant(5)

# Launch the default graph.
with tf.Session() as sess:
    print("c1=4, c2=5")
    print("Addition with constants: %i" % sess.run(c1+c2))
    print("Multiplication with constants: %i" % sess.run(c1*c2))

# Basic Operations with variable as graph input
# The value returned by the constructor represents the output
# of the Variable op. (define as input when running session)
# tf Graph input
c1 = tf.placeholder(tf.int16)
c2 = tf.placeholder(tf.int16)

# Define some operations
add = tf.add(c1, c2)
mul = tf.multiply(c1, c2)

# Launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    print("Addition with variables: %i" % sess.run(add, feed_dict={c1: 4, c2: 5}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={c1: 4, c2: 5}))

