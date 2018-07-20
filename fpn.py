import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

SAVE_SESS = True;
SHOW_PLOT = False;
KEEP_PROB = 0.75;
LEARNING_RATE = 1e-2;
ITERATION_NUM = 10;

class conv7x7(object):
    def __init__(self,x,stride=1,c1=None,c2=None,BN=True,p=1):
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
            self.out = tf.nn.dropout(tf.clip_by_value(aft_bn, 0, 1),p);
        else:
            self.out = tf.nn.dropout(net_out,p);
            
class conv1x1(object):
    def __init__(self,x,c1=None,c2=None,BN=False,p=1):
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
            self.out = tf.nn.dropout(tf.clip_by_value(aft_bn, 0, 1),p);
        else:
            self.out = tf.nn.dropout(net_out,p);

        
def regen_feature(x,channel=None,depth=None):
    pad = 3*(np.power(2,depth)-1);
    scale = tf.cast(np.power(2.,depth),tf.int32);
    paddings = tf.constant([[0,0],[pad,pad],[pad,pad]]);
    map_small = conv1x1(x,channel,1,False,KEEP_PROB);
    map_large_0 = tf.reshape(map_small.out,[-1,tf.shape(map_small.out)[1],1,tf.shape(map_small.out)[2],1]);
    map_large_1 = tf.tile(map_large_0,[1,1,scale,1,scale]);
    map_large_2 = tf.reshape(map_large_1,[-1,tf.multiply(tf.shape(map_small.out)[1],scale),tf.multiply(tf.shape(map_small.out)[1],scale)]);    
    map_pad = tf.pad(map_large_2,paddings,"CONSTANT");
    return map_pad;

iteration_count = tf.Variable(0, dtype=tf.int64)
iteration_add = tf.assign(iteration_count,iteration_count+ITERATION_NUM);

files = ['res/img/' + s for s in sorted(os.listdir('res/img/'))];
files_gt = ['res/ground_truth/' + s for s in sorted(os.listdir('res/ground_truth/'))];
img_raw = tf.stack([tf.read_file(file) for file in files]);
img_raw_gt = tf.stack([tf.read_file(file) for file in files_gt]);
img_decoded = tf.cast(tf.map_fn(fn = tf.image.decode_jpeg, elems = img_raw, dtype=tf.uint8),tf.float32)/256;
img_decoded_gt = tf.cast(tf.map_fn(fn = tf.image.decode_jpeg, elems = img_raw_gt, dtype=tf.uint8),tf.float32)/256;
img_gt = tf.reshape(img_decoded_gt,[tf.shape(img_decoded_gt)[0],tf.shape(img_decoded_gt)[1],tf.shape(img_decoded_gt)[2]]);


layer1 = conv7x7(img_decoded,2,3,256,True,KEEP_PROB);
feature1 = regen_feature(layer1.out,256,1);
layer2 = conv7x7(layer1.out,2,256,256,True,KEEP_PROB);
feature2 = regen_feature(layer2.out,256,2);
layer3 = conv7x7(layer2.out,2,256,256,True,KEEP_PROB);
feature3 = regen_feature(layer3.out,256,3);
layer4 = conv7x7(layer3.out,2,256,256,True,KEEP_PROB);
feature4 = regen_feature(layer4.out,256,4);
feature = feature1+feature2+feature3+feature4;

cost = tf.reduce_sum(tf.squared_difference(img_gt, feature));
opt = tf.train.AdamOptimizer(LEARNING_RATE);
train_op = opt.minimize(cost);

sess = tf.Session();
sess.run(tf.global_variables_initializer());
saver = tf.train.Saver();

if SAVE_SESS is True:
    try:
        saver.restore(sess,os.getcwd()+'/save/save.ckpt');
        print('checkpoint loaded');
    except:
        print('cannot load checkpoint');
    
    saver.save(sess,os.getcwd()+'/save/save.ckpt');



if SHOW_PLOT:
    plt.ion();
    fig = plt.figure(figsize=(10, 5));
    p1 = fig.add_subplot(1, 2, 1);
    p2 = fig.add_subplot(1, 2, 2);

i = 0;
while(True):
    if i%ITERATION_NUM == 0:
        loss = sess.run(cost);
        print("iteration ",repr(sess.run(iteration_count)),", loss is ",repr(loss));
        sess.run(iteration_add);
        saver.save(sess,os.getcwd()+'/save/save.ckpt');
        if SHOW_PLOT:
            gt = sess.run(img_gt);
            out_f = sess.run(feature);
            p1.imshow(gt[0]);
            p2.imshow(out_f[0]);
            fig.canvas.draw()
            plt.pause(0.1)
    sess.run(train_op);
    
    i = i+1;

