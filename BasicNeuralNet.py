import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#Tutorial found on http://adventuresinmachinelearning.com/python-tensorflow-tutorial/

#Import Data
mnist = input_data.read_data_sets("MNUST_data/", one_hot=True)

# Vars
learning_rate = 0.5
epochs = 10
batch_size = 100

#declare the training data placeholders
#input x - for 28 x 28 pixels = 784

x = tf.placeholder(tf.float32, [None, 784])

#now declare the output data placeholder - 10 digits

y = tf.placeholder(tf.float32, [None, 10])

# now declare the weights connected the inputs to the hidden layer
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')

# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10], name='b2'))

# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
# relu finds the max of the matrix
hidden_out = tf.nn.relu(hidden_out)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer

y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))


# add an optimizer

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)


# Finally setup the initalization  operator

init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# add a summary to store the accuracy
tf.summary.scalar('accuracy', accuracy)

# Update and store summary
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./Tutorials/Tensorflow/NeuralNet/graphs')


# Start the session
with tf.Session() as sess:

    # init the vars
    sess.run(init_op)

    total_batch = int(len(mnist.train.labels) / batch_size)
    
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            # Get next data set
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            # Run it through the network, and then optimize it based on result
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            # Calculate the average cost
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost))
        summary = sess.run(merged, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        writer.add_summary(summary, epoch)

    print("\nTraining complete!")
    writer.add_graph(sess.graph)
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))