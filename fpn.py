import os
import tensorflow as tf
import matplotlib.pyplot as plt

files = ['res/' + str for str in sorted(os.listdir('res/'))];
img_raw = tf.stack([tf.read_file(file) for file in files]);
img_decoded = tf.cast(tf.map_fn(fn = tf.image.decode_jpeg, elems = img_raw, dtype=tf.uint8),tf.float32)/256;

sess = tf.Session();
image = sess.run(img_decoded);
print(image);

plt.imshow(image[0]);
plt.show();