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
            self.out = net_out;
            
class conv1x1(object):
    def __init__(self,x,c1=None,c2=None,BN=False):
        w = tf.Variable(tf.random_normal(shape=[1,1,c1,c2],stddev=1));
        b = tf.Variable(tf.zeros(c2));
        conv_out = tf.nn.conv2d(input = x, filter = w, strides = [1,1,1,1], padding = 'VALID');
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
            self.out = net_out;

        
def regen_feature(x,channel=None,depth=None):
    pad = 3*(np.power(2,depth)-1);
    scale = tf.cast(np.power(2.,depth),tf.int32);
    paddings = tf.constant([[0,0],[pad,pad],[pad,pad]]);
    map_small = conv1x1(x,channel,1,True);
    map_large_0 = tf.reshape(map_small.out,[-1,tf.shape(map_small.out)[1],1,tf.shape(map_small.out)[2],1]);
    map_large_1 = tf.tile(map_large_0,[1,1,scale,1,scale]);
    map_large_2 = tf.reshape(map_large_1,[-1,tf.multiply(tf.shape(map_small.out)[1],scale),tf.multiply(tf.shape(map_small.out)[1],scale)]);    
    map_pad = tf.pad(map_large_2,paddings,"CONSTANT");
    return map_pad;

files = ['res/' + s for s in sorted(os.listdir('res/'))];
img_raw = tf.stack([tf.read_file(file) for file in files]);
img_decoded = tf.cast(tf.map_fn(fn = tf.image.decode_jpeg, elems = img_raw, dtype=tf.uint8),tf.float32)/256;

layer1 = conv7x7(img_decoded,2,3,256,True);
feature1 = regen_feature(layer1.out,256,1);
layer2 = conv7x7(layer1.out,2,256,256,True);
feature2 = regen_feature(layer2.out,256,2);
layer3 = conv7x7(layer2.out,2,256,256,True);
feature3 = regen_feature(layer3.out,256,3);
layer4 = conv7x7(layer3.out,2,256,256,True);
feature4 = regen_feature(layer4.out,256,4);
feature = (feature1+feature2+feature3+feature4)/4;

layer_out = tf.reshape(layer4.out,[-1,7,7]);

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

convout = np.array(sess.run(feature));
convin = np.array(sess.run(img_decoded));
print(convout);
print(np.shape(convin));
print(np.shape(convout));
print(np.shape(sess.run(layer1.out)));
print(np.shape(sess.run(feature1)));
print(np.shape(sess.run(layer2.out)));
print(np.shape(sess.run(feature2)));
print(np.shape(sess.run(layer3.out)));
print(np.shape(sess.run(feature3)));
print(np.shape(sess.run(layer4.out)));
print(np.shape(sess.run(feature4)));
plt.figure(1);
plt.hist(convout.flatten(),100);
plt.figure(2);
plt.hist(convin.flatten(),100);
plt.figure(3);
plt.imshow(convout[0]);
plt.figure(4);
plt.imshow(convin[0]);
plt.show();