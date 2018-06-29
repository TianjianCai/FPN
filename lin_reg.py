"""
A linear regression program using tensorflow
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = tf.placeholder(tf.float32);
y = tf.placeholder(tf.float32);

k = tf.Variable(1.,tf.float32);
b = tf.Variable(1.,tf.float32);
y_out = tf.add(tf.multiply(x, k),b);

loss = tf.reduce_sum(tf.square(tf.subtract(y, y_out)));

opt = tf.train.GradientDescentOptimizer(learning_rate=1e-5);
train_op = opt.minimize(loss);

sample_k = 3;
sample_b = -0.9;
noise = 3*np.random.randn(1000);
sample_x = np.linspace(start = 0, stop = 10, num = 1000);
sample_y = sample_x*sample_k+sample_b+noise;

print("session started");
sess = tf.Session();
sess.run(tf.global_variables_initializer());

i=0;
delta = np.Infinity;
cost_old = np.Infinity;
while delta > 1e-5:
    i = i+1;
    sess.run(train_op,{x:sample_x,y:sample_y});
    cost = sess.run(loss,{x:sample_x,y:sample_y});
    delta = np.abs(cost_old - cost);
    cost_old = cost;
    print("epoch %d, cost = %f, delta = %f" % (i,cost,delta));

k_out = sess.run(k);
b_out = sess.run(b);
x2 = np.linspace(0,10);
y2 = x2*k_out + b_out;
print("k = %f" % (k_out));
print("b = %f" % (b_out));

plt.scatter(x = sample_x, y = sample_y, c = 'y', marker = '.');
plt.plot(x2,y2);
plt.show();