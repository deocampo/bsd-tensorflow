'''
Created on Jul 19, 2017

@author: deocampo
'''


import tensorflow as tf
import numpy as np
from numpy import matrix

matrix1 = [[5.0, 4.0], 
           [3.0, 2.0]]

matrix2 = np.array([[4.0, 5.0], 
                    [6.0, 7.0]], dtype=np.float32)

matrix3 = tf.constant([[6.0, 7.0], 
                        [8.0, 9.0]])


print(matrix1)
print(type(matrix1))
print(matrix2)
print(type(matrix2))
print(matrix3)
print(type(matrix3))


matrixA = tf.convert_to_tensor(matrix1, dtype=tf.float32)
matrixB = tf.convert_to_tensor(matrix2, dtype=tf.float32)
matrixC = tf.convert_to_tensor(matrix3, dtype=tf.float32)

print(matrixA)
print(type(matrixA))
print(matrixB)
print(type(matrixB))
print(matrixC)
print(type(matrixC))

