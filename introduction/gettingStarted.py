import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

# introduction to 'Session'
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# introduction to 'Constant'
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
print(sess.run([node1, node2]))

# introduction to 'add' operatoins 
node3 = tf.add(node1, node2)
print("node3 - ", node3)
print("sess.run(node3) - ",sess.run(node3))

# introduction to 'placeholder'
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
print("sess.run(adder_node, {a: 3, b:4.5}) - ", sess.run(adder_node, {a: 3, b:4.5}))
print("sess.run(adder_node, {a: [1,3], b: [2, 4]}) - ", sess.run(adder_node, {a: [1,3], b: [2, 4]}))

# introduction to more complex operations
add_and_triple = adder_node * 3.
print("sess.run(add_and_triple, {a: 3, b:4.5}) - ", sess.run(add_and_triple, {a: 3, b:4.5}))


# introduction to 'Variables'
W = tf.Variable([.3], tf.float32)
U = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + U

init = tf.global_variables_initializer()
sess.run(init)
print("sess.run(linear_model, {x: [1,2,3,4]})", sess.run(linear_model, {x: [1,2,3,4]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

fixW = tf.assign(W, [-1.])
fixU = tf.assign(U, [1.])
sess.run([fixW, fixU])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))


