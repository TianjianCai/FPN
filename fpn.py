import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class conv3x3(object):
    def __init__(self,x,stride=1,c1=None,c2=None):
        w1 = tf.Variable(tf.random_normal(shape=[1,3,c1,c1]));
        w2 = tf.Variable(tf.random_normal(shape=[3,1,c1,c2]));
        b = tf.Variable(tf.zeros(c2));
        mid = tf.nn.conv2d(input = x, filter = w1, strides = [1,1,stride,1], padding = 'VALID');
        fin = tf.nn.conv2d(input = mid, filter = w2, strides = [1,stride,1,1], padding = 'VALID');
        self.out = tf.nn.bias_add(fin, b);

files = ['res/' + s for s in sorted(os.listdir('res/'))];
img_raw = tf.stack([tf.read_file(file) for file in files]);
img_decoded = tf.cast(tf.map_fn(fn = tf.image.decode_jpeg, elems = img_raw, dtype=tf.uint8),tf.float32)/256;

layer1 = conv3x3(img_decoded,1,3,256);
layer2 = conv3x3(layer1.out,1,256,256)
layer3 = conv3x3(layer2.out,2,256,3)

sess = tf.Session();
sess.run(tf.global_variables_initializer());

convout = sess.run(layer3.out)
print(convout);
print(np.shape(convout))

plt.imshow(convout[0]);
plt.show();