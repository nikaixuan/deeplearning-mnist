import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tempfile
import pandas as pd
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator




def trans_to_array(label):
    test = np.zeros((label.shape[0],10))
    for i in range(test.shape[0]):
        test[i,int(label[i,0])] = 1
    return test


def scale(images):
    return np.multiply(images.astype(np.float32), 1.0 / 255.0)

data = np.load('data.npz')
num_classes = 10
nchannels,rows,cols = 1,64,64
rawfeature = data['train_X']
features = scale(rawfeature)
labels = trans_to_array(data['train_y'][:,np.newaxis])
x_test = scale(data['test_X'])

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


DEPTH   = 3                 # Depth of a tree
N_LEAF  = 2 ** (DEPTH + 1)  # Number of leaf node
N_LABEL = 10                # Number of classes
N_TREE  = 5                 # Number of trees (ensemble)
N_BATCH = 50               # Number of data points per mini-batch

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_bias(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_prob_weights(shape, minval=-5, maxval=5):
    return tf.Variable(tf.random_uniform(shape, minval, maxval))

def model(X, w, w2, w3, w4_e, b, b2, b3, b4_e, w_d_e, w_l_e, p_keep_conv, p_keep_hidden):
    """
    Create a forest and return the neural decision forest outputs:
        decision_p_e: decision node routing probability for all ensemble
            If we number all nodes in the tree sequentially from top to bottom,
            left to right, decision_p contains
            [d(0), d(1), d(2), ..., d(2^n - 2)] where d(1) is the probability
            of going left at the root node, d(2) is that of the left child of
            the root node.
            decision_p_e is the concatenation of all tree decision_p's
        leaf_p_e: terminal node probability distributions for all ensemble. The
            indexing is the same as that of decision_p_e.
    """
    assert(len(w4_e) == len(w_d_e))
    assert(len(w4_e) == len(w_l_e))

    l1a = tf.nn.relu(tf.nn.conv2d(X, w, [1, 1, 1, 1], 'SAME')+b)
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME')+b2)
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME')+b3)
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    l3 = tf.reshape(l3, [-1, w4_e[0].get_shape().as_list()[0]])

    decision_p_e = []
    leaf_p_e = []
    for w4, b4, w_d, w_l in zip(w4_e, b4_e, w_d_e, w_l_e):
        l4 = tf.nn.relu(tf.matmul(l3, w4)+b4)
        l4 = tf.nn.dropout(l4, p_keep_hidden)

        decision_p = tf.nn.sigmoid(tf.matmul(l4, w_d))
        leaf_p = tf.nn.softmax(w_l)

        decision_p_e.append(decision_p)
        leaf_p_e.append(leaf_p)

    return decision_p_e, leaf_p_e


##################################################
# Initialize network weights
##################################################
w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
b = init_bias([32])
b2 = init_bias([64])
b3 = init_bias([128])

w4_ensemble = []
b4_ensemble = []
w_d_ensemble = []
w_l_ensemble = []
for i in range(N_TREE):
    w4_ensemble.append(init_weights([128 * 8 * 8, 1024]))
    b4_ensemble.append(init_bias([1024]))
    w_d_ensemble.append(init_prob_weights([1024, N_LEAF], -1, 1))
    w_l_ensemble.append(init_prob_weights([N_LEAF, N_LABEL], -2, 2))

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")


##################################################
# Define a fully differentiable deep-ndf
##################################################
# With the probability decision_p, route a sample to the right branch
X = tf.placeholder(tf.float32, [N_BATCH, 64, 64, 1])
Y = tf.placeholder(tf.float32, [N_BATCH, N_LABEL])
decision_p_e, leaf_p_e = model(X, w, w2, w3, w4_ensemble, b, b2, b3, b4_ensemble, w_d_ensemble,
                               w_l_ensemble, p_keep_conv, p_keep_hidden)

flat_decision_p_e = []

# iterate over each tree
for decision_p in decision_p_e:
    # Compute the complement of d, which is 1 - d
    # where d is the sigmoid of fully connected output
    decision_p_comp = tf.subtract(tf.ones_like(decision_p), decision_p)

    # Concatenate both d, 1-d
    decision_p_pack = tf.stack([decision_p, decision_p_comp])

    # Flatten/vectorize the decision probabilities for efficient indexing
    flat_decision_p = tf.reshape(decision_p_pack, [-1])
    flat_decision_p_e.append(flat_decision_p)

# 0 index of each data instance in a mini-batch
batch_0_indices = \
    tf.tile(tf.expand_dims(tf.range(0, N_BATCH * N_LEAF, N_LEAF), 1),
            [1, N_LEAF])


###############################################################################
# The routing probability computation
#
# We will create a routing probability matrix \mu. First, we will initialize
# \mu using the root node d, 1-d. To efficiently implement this routing, we
# will create a giant vector (matrix) that contains all d and 1-d from all
# decision nodes. The matrix version of that is decision_p_pack and vectorized
# version is flat_decision_p.
#
# The suffix `_e` indicates an ensemble. i.e. concatenation of all responsens
# from trees.
#
# For depth = 2 tree, the routing probability for each leaf node can be easily
# compute by multiplying the following vectors elementwise.
# \mu =       [d_0,   d_0,   d_0,   d_0, 1-d_0, 1-d_0, 1-d_0, 1-d_0]
# \mu = \mu * [d_1,   d_1, 1-d_1, 1-d_1,   d_2,   d_2, 1-d_2, 1-d_2]
# \mu = \mu * [d_3, 1-d_3,   d_4, 1-d_4,   d_5, 1-d_5,   d_6, 1-d_6]
#
# Tree indexing
#      0
#    1   2
#   3 4 5 6
##############################################################################
in_repeat = N_LEAF / 2
out_repeat = N_BATCH

# Let N_BATCH * N_LEAF be N_D. flat_decision_p[N_D] will return 1-d of the
# first root node in the first tree.
batch_complement_indices = \
    np.array([[0] * int(in_repeat), [N_BATCH * N_LEAF] * int(in_repeat)]
             * out_repeat).reshape(N_BATCH, N_LEAF)

# First define the routing probabilities d for root nodes
mu_e = []

# iterate over each tree
for i, flat_decision_p in enumerate(flat_decision_p_e):
    mu = tf.gather(flat_decision_p,
                   tf.add(batch_0_indices, batch_complement_indices))
    mu_e.append(mu)

# from the second layer to the last layer, we make the decision nodes
for d in range(1, DEPTH + 1):
    indices = tf.range(2 ** d, 2 ** (d + 1)) - 1
    tile_indices = tf.reshape(tf.tile(tf.expand_dims(indices, 1),
                                      [1, 2 ** (DEPTH - d + 1)]), [1, -1])
    batch_indices = tf.add(batch_0_indices, tf.tile(tile_indices, [N_BATCH, 1]))

    in_repeat = in_repeat / 2
    out_repeat = out_repeat * 2

    # Again define the indices that picks d and 1-d for the node
    batch_complement_indices = \
        np.array([[0] * int(in_repeat), [N_BATCH * N_LEAF] * int(in_repeat)]
                 * int(out_repeat)).reshape(N_BATCH, N_LEAF)

    mu_e_update = []
    for mu, flat_decision_p in zip(mu_e, flat_decision_p_e):
        mu = tf.multiply(mu, tf.gather(flat_decision_p,
                                  tf.add(batch_indices, batch_complement_indices)))
        mu_e_update.append(mu)

    mu_e = mu_e_update

##################################################
# Define p(y|x)
##################################################
py_x_e = []
for mu, leaf_p in zip(mu_e, leaf_p_e):
    # average all the leaf p
    py_x_tree = tf.reduce_mean(
        tf.multiply(tf.tile(tf.expand_dims(mu, 2), [1, 1, N_LABEL]),
               tf.tile(tf.expand_dims(leaf_p, 0), [N_BATCH, 1, 1])), 1)
    py_x_e.append(py_x_tree)

py_x_e = tf.stack(py_x_e)
py_x = tf.reduce_mean(py_x_e, 0)


features_new = features.reshape(-1, 64, 64, 1)
test_new = x_test.reshape(-1, 64, 64, 1)
# cross entropy loss
cost = tf.reduce_mean(-tf.multiply(tf.log(py_x), Y))

# cost = tf.reduce_mean(tf.nn.cross_entropy_with_logits(py_x, Y))
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict = tf.argmax(py_x, 1)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(50):
    # One epoch
    for start, end in zip(range(0, len(features_new)+1, N_BATCH), range(N_BATCH, len(features_new)+1, N_BATCH)):
        sess.run(train_step, feed_dict={X: features_new[start:end], Y: labels[start:end],
                                        p_keep_conv: 0.8, p_keep_hidden: 0.5})
    print(i)

result = []
for start_t, end_t in zip(range(0, len(test_new)+1, N_BATCH), range(N_BATCH, len(test_new)+1, N_BATCH)):
        result.extend(sess.run(predict, feed_dict={X: test_new[start_t:end_t],p_keep_conv: 1.0,
                                         p_keep_hidden: 1.0}))

print(result)
new_result = []
for i in range(len(result)):
    new_result.append((i+1,result[i]))
lab = ['Id','Label']
df = pd.DataFrame.from_records(new_result, columns=lab)
print(new_result)
df.to_csv('out2.csv',index=False,header=True)