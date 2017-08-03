from tensorflow.examples.tutorials.mnist import input_data

#Get The MNIST Data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# Create the model
inputs = tf.placeholder(tf.float32, [None, 784])
weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))
predictions = tf.matmul(inputs, weights) + biases

# Define loss and optimizer
expectedPredictions = tf.placeholder(tf.float32, [None, 10])

# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.

meanError = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=expectedPredictions, logits=predictions))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(meanError)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={inputs: batch_xs, expectedPredictions: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(expectedPredictions, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={inputs: mnist.test.images,
                                        expectedPredictions: mnist.test.labels}))