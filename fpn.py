import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

SAVE_SESS = True;

class conv3x3(object):
    def __init__(self,x,stride=1,c1=None,c2=None):
        w1 = tf.Variable(tf.random_normal(shape=[1,3,c1,c1],stddev=0.1));
        w2 = tf.Variable(tf.random_normal(shape=[3,1,c1,c2],stddev=0.1));
        b = tf.Variable(tf.zeros(c2));
        beta = tf.Variable(tf.zeros(c2));
        gamma = tf.Variable(tf.ones(c2));
        mid = tf.nn.conv2d(input = x, filter = w1, strides = [1,1,stride,1], padding = 'VALID');
        fin = tf.nn.conv2d(input = mid, filter = w2, strides = [1,stride,1,1], padding = 'VALID');
        bef_bn = tf.nn.bias_add(fin, b);
        aft_bn,_,_ = tf.nn.fused_batch_norm(
            x=bef_bn, 
            scale=gamma, 
            offset=beta, 
            is_training=True);
        self.out = aft_bn;

files = ['res/' + s for s in sorted(os.listdir('res/'))];
img_raw = tf.stack([tf.read_file(file) for file in files]);
img_decoded = tf.cast(tf.map_fn(fn = tf.image.decode_jpeg, elems = img_raw, dtype=tf.uint8),tf.float32)/256;

layer1 = conv3x3(img_decoded,1,3,256);
layer2 = conv3x3(layer1.out,1,256,256);
layer3 = conv3x3(layer2.out,2,256,3);

sess = tf.Session();
sess.run(tf.global_variables_initializer());

if SAVE_SESS is True:
    saver = tf.train.Saver();
    try:
        saver.restore(sess,os.getcwd()+'/save/save.ckpt');
        print('checkpoint loaded');
    except:
        print('cannot load checkpoint');
    
    saver.save(sess,os.getcwd()+'/save/save.ckpt');


convout = np.array(sess.run(layer3.out));
convin = np.array(sess.run(img_decoded));
print(convout);
print(np.shape(convin));
print(np.shape(convout));
plt.figure(1);
plt.hist(convout.flatten(),100);
plt.figure(2);
plt.hist(convin.flatten(),100);
plt.figure(3);
plt.imshow(convout[0]);
plt.figure(4);
plt.imshow(convin[0]);
plt.show();