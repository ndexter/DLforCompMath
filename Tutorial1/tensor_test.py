import os
import time
import tensorflow as tf
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print('Running tensorflow with version:')
print(tf.__version__)

tf.enable_eager_execution()

tf.executing_eagerly()

x_data = np.array([[[ 1.,  2.,  3.], [ 4.,  5.,  6.]],
                   [[ 7.,  8.,  9.], [10., 11., 12.]],
                   [[13., 14., 15.], [16., 17., 18.]]])

print(x_data)

x = tf.convert_to_tensor(x_data, dtype = tf.float32)

print(x)

print(tf.slice(x,[0,0,0],[3,2,3])) # the whole tensor
print(tf.slice(x,[1,0,0],[2,2,3])) 
print(tf.slice(x,[2,0,0],[1,2,3])) 
print(tf.slice(x,[0,1,0],[3,1,3]))
print(tf.slice(x,[0,0,1],[3,2,1]))
print(tf.slice(x,[0,0,1],[3,2,2]))

x_data = np.array([[1,2,3],[4,5,6],[7,8,900]])

print(x_data)

x = tf.convert_to_tensor(x_data, dtype = tf.float32)

print(x)

print(tf.slice(x,[0,0],[3,3])) # the whole tensor
print(tf.slice(x,[0,0],[3,2])) # 2 cols from first col
print(tf.slice(x,[0,1],[3,2])) # 2 cols from 2nd col
print(tf.slice(x,[0,0],[3,1])) # first col
print(tf.slice(x,[0,1],[3,1])) # 2nd col

print('\n\n x\'s 3rd column is')
print(tf.slice(x,[0,2],[3,1])) # 3rd col

print('\n\n')
print('computing the scalar product of matrix x with its 3rd col')
print(x*tf.slice(x,[0,2],[3,1])) # scalar product of the elements of matrix x with its 3rd col
print('computing the scalar product of matrix x with its 3rd col, then add 3rd col')
print(x*tf.slice(x,[0,2],[3,1]) + tf.slice(x,[0,2],[3,1])) # scalar product of the elements of matrix x with its 3rd col

print(x)

print('matvec product of matrix x with its 3rd col')
print(tf.matmul(x,tf.slice(x,[0,2],[3,1]))) # matvec product of matrix x with its 3rd col

print('matvec product of matrix x with its 3rd col, then add 3rd col')
print(tf.matmul(x,tf.slice(x,[0,2],[3,1])) + tf.slice(x,[0,2],[3,1])) # matvec product of matrix x with its 3rd col

z = tf.slice(x,[0,2],[3,1])

print(tf.matmul(z,tf.transpose(z)))


print('x = ')
print(x)

print('max of x is ')
mx = tf.reduce_max(x)
print(mx.numpy())


k = 300

W_fr_data = np.random.rand(k,k);
w_data = np.random.rand(k,1);
X_data = np.random.rand(k,150);

W_fr = tf.convert_to_tensor(W_fr_data, dtype = tf.float32)
w = tf.convert_to_tensor(w_data, dtype = tf.float32)
X = tf.convert_to_tensor(X_data, dtype = tf.float32)

#print('w = ')
#print(w)
#print('X = ')
#print(X)

print('w shape = ')
print(w.shape)
print('W_fr shape = ')
print(W_fr.shape)
print('X shape = ')
print(X.shape)

nb_iter_test = 50000;

start_time = time.time()

#wT = tf.transpose(w)

for i in range(nb_iter_test):
    z = tf.matmul(W_fr,X)

diff_time = time.time() - start_time

print('Case 0: z = W*X (W full rank) took ' + str(diff_time) + ' seconds')

start_time = time.time()

#wT = tf.transpose(w)

for i in range(nb_iter_test):
    W = tf.matmul(w, tf.transpose(w))
    z = tf.matmul(W,X)

diff_time = time.time() - start_time

print('Case 1: z = W*X took ' + str(diff_time) + ' seconds')

start_time = time.time()

for i in range(nb_iter_test):
    c = tf.tensordot(tf.transpose(w),X,1)
    z = c*w

diff_time = time.time() - start_time

print('Case 2: z = w^T*X*w took ' + str(diff_time) + ' seconds')

#print('c = ')
#print(c)

#print('c shape = ')
#print(c.shape)
print('z shape = ')
print(z.shape)
#print('z = ')
#print(z)

diff_time = time.time() - start_time

