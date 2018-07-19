import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

SAVE_SESS = True;

class conv7x7(object):
    def __init__(self,x,stride=1,c1=None,c2=None,BN=True):
        w = tf.Variable(tf.random_normal(shape=[7,7,c1,c2],stddev=1));
        b = tf.Variable(tf.zeros(c2));
        conv_out = tf.nn.conv2d(input = x, filter = w, strides = [1,stride,stride,1], padding = 'VALID');
        net_out = tf.nn.bias_add(conv_out, b);
        
        beta = tf.Variable(0.5*tf.ones(c2));
        gamma = tf.Variable(0.25*tf.ones(c2));
        aft_bn,_,_ = tf.nn.fused_batch_norm(
            x=net_out, 
            scale=gamma, 
            offset=beta, 
            is_training=True);
        if BN:
            self.out = tf.clip_by_value(aft_bn, 0, 1);
        else:
            self.out = aft_bn;


files = ['res/' + s for s in sorted(os.listdir('res/'))];
img_raw = tf.stack([tf.read_file(file) for file in files]);
img_decoded = tf.cast(tf.map_fn(fn = tf.image.decode_jpeg, elems = img_raw, dtype=tf.uint8),tf.float32)/256;

layer1 = conv7x7(img_decoded,2,3,256,False);
layer2 = conv7x7(layer1.out,2,256,3,False);

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

convout = np.array(sess.run(layer2.out));
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