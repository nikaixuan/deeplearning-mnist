import numpy as np
import tensorflow as tf
import tempfile
import pandas as pd

def trans_to_array(label):
    test = np.zeros((label.shape[0],10))
    for i in range(test.shape[0]):
        test[i,int(label[i,0])] = 1
    return test

def scale(images):
    return np.multiply(images.astype(np.float32), 1.0 / 255.0)


data = np.load('data.npz')
num_classes = 10
nchannels, rows, cols = 1, 64, 64
rawfeature = data['train_X']
features = scale(rawfeature)
labels = trans_to_array(data['train_y'][:, np.newaxis])

x_test = scale(data['test_X'])

print(labels)
# print ("Information on dataset")
# print ("x_train", x_train.shape)
# print ("targets_train", targets_train.shape)
#
# print ("x_test", x_test.shape)
# print ("targets_test", targets_test.shape)

def next_batch(index,feature,label,batch_size):
    """Return the next `batch_size` examples from this data set."""
    epochs_completed = 0
    examples = feature.shape[0]
    start = index*batch_size
    index_in_epoch =index*batch_size+batch_size-1
    if index_in_epoch > examples:
      # Finished epoch
      epochs_completed += 1
      # Shuffle the data
      perm = np.arange(examples)
      np.random.shuffle(perm)
      feature = feature[perm]
      label = label[perm]
      # Start next epoch
      start = 0
      index_in_epoch = batch_size
      assert batch_size <= examples
    end = index_in_epoch
    return feature[start:end], label[start:end]


def deepnn(x):
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 64, 64, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

        # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    # Second pooling layer.
    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([3, 3, 128, 256])
        b_conv4 = bias_variable([256])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

    # Second pooling layer.
    with tf.name_scope('pool4'):
        h_pool4 = max_pool_2x2(h_conv4)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([4 * 4 * 256, 1024])
        b_fc1 = bias_variable([1024])

        h_pool4_flat = tf.reshape(h_pool4, [-1, 4 * 4 * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Create the model
x = tf.placeholder(tf.float32, [None, 4096])

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

# Build the graph for the deep net
y_conv, keep_prob = deepnn(x)

with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                        logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

graph_location = tempfile.mkdtemp()
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch = next_batch(i,features,labels,50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    feed_dict = {x: x_test,keep_prob:1.0}
    classification = sess.run(y_conv,feed_dict)
    print(classification)
    class_tensor = tf.argmax(classification,1)
    result = sess.run(class_tensor)
    print(result)
    new_result = []
    for i in range(len(result)):
        new_result.append((i+1,result[i]))
    lab = ['Id','Label']
    df = pd.DataFrame.from_records(new_result, columns=lab)
    print(new_result)
    df.to_csv('out2.csv',index=False,header=True)
    # print('test accuracy %g' % accuracy.eval(feed_dict={
    #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))