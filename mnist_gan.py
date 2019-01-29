import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/h4ckfu/Code/test/TheGANs/MNIST_data/", one_hot=True)

################### All your functions are belong... ##########################

# http://bit.ly/leaky_relu
def my_leaky_relu(x):
    '''
    Basically choooses the max(x, x * 0.01) as an activation function
    '''
    return tf.nn.leaky_relu(x, alpha=0.01)

### The G ###
# the z is going to be the random noise that we start with...
def generator(z, reuse=None):

    with tf.variable_scope('gen', reuse=reuse):

        hidden1 = tf.layers.dense(inputs = z, units=128, activation=my_leaky_relu)
        hidden2 = tf.layers.dense(inputs = hidden1, units = 128, activation=my_leaky_relu)

        # 784, of course, is the flatened 28x28 pixel image we want to generate
        # tanh - NOT 0 to 1 it's -1 to 1 -- so random noise will be -1 - 1 as well
        output = tf.layers.dense(hidden2, units = 784, activation = tf.nn.tanh)

        return output


### The D ###
def discriminator(X, reuse=None):

    with tf.variable_scope('dis', reuse=reuse):

        hidden1 = tf.layers.dense(inputs = X, units = 128, activation=my_leaky_relu)
        hidden2 = tf.layers.dense(inputs = hidden1, units = 128, activation=my_leaky_relu)

        logits = tf.layers.dense(hidden2, units = 1) # probability of real or fake
        output = tf.sigmoid(logits) # sigmoid of the logits

        return output , logits


### Helper function for calculating loss with cross entropy (with_logits)
def loss_func(logits_in, labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits = logits_in, labels=labels_in))


### TensorFlow placeholders
# Places in memory where we'll store values later when defing these in the feed_dict

# shape =  None (batch size, so any number of rows) x 784 pixels (columns)
real_images = tf.placeholder(tf.float32, shape=[None, 784])

# What we're feeding our generator - ,100 random points, its the holder for the noise
z = tf.placeholder(tf.float32, shape=[None, 100])

############################ Variables and such ###############################

learning_rate = 0.001

batch_size = 100

label_smothing = 0.9

# 100 times to low - just here so I can test to make sure script still works
epochs = 5

# this is the list that will hold sample images when we run the session
samples = []


############### Calling the functions (cleaned up) #############################

# Feeding G the placeholder z which we'll pass in as noise with the feed_dict in sess
G = generator(z)

# Feed D real images (first) for training so D knows what the img's should look like
D_output_real, D_logits_real = discriminator(real_images)

# Feed discriminator generated fakes by by passing in the results from G(z)
# *** NOTE: This is the first way the functions interact
D_output_fake, D_logits_fake = discriminator(G, reuse=True)


############### Using the trainable_variables from the Functions ###############

# list of tf.Variable created via the layers api  in the functions
tvars = tf.trainable_variables()

# List comprehension - works because tf.variable_scope in the G & D functions
# Gonna use these with AdamOptimizer
d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]

########################### Calculating loss ##################################

# loss_func is essentially tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits)
# *** NOTE: G_loss = loss_func(D_logits_fake) is the 2nd way D & G connect...

# We want all of the real data labels to be true - hense the tf.ones as labels
# Because = real_images But not exactly 1 as to not overfit - "close to 1" = smooting
D_real_loss = loss_func(D_logits_real, tf.ones_like(D_logits_real) * label_smothing)

# so, just the opposite - all the labels are zero cause they are all fakes
D_fake_loss = loss_func(D_logits_fake, tf.zeros_like(D_logits_fake))

# Final discriminator and generator loss
D_loss = D_real_loss + D_fake_loss

# Remember D_logits_fake is actually discriminator(G) so logits(G -> D) - labels = 1
# the generator "thinks" its making true images so labels = 1
G_loss = loss_func(D_logits_fake, tf.ones_like(D_logits_fake))


######## Calling Adam Optimizer train the minimized loss on these lists of variables

# Minimize the D_loss on the d_var list ( of discriminator variables)
D_trainer = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=d_vars)
G_trainer = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=g_vars)

########################### The Session Then ###################################

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for epoch in range(epochs):

        # how many batches does it take to go through all the training data
        num_batches = mnist.train.num_examples // batch_size

        for i in range(num_batches):

            batch = mnist.train.next_batch(batch_size)

            batch_images = batch[0].reshape((batch_size, 784))

            # rescale for the tanh activation function
            batch_images = batch_images * 2 - 1

            # again -1 to 1 because tanh - 100 is for the 100 random points
            batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))

            # Run the optimizers - only care about the generators output tho
            # This is running the output of AdamOptimizer -> minimize the loss of the D variables
            _ = sess.run(D_trainer, feed_dict={real_images:batch_images, z:batch_z})
            _ = sess.run(G_trainer, feed_dict={z:batch_z})

        print("ON EPOCH {}".format(epoch))

        # Sample from the Generator as we are training
        # Pass in some noise size in the shape of: 1 image, 100(batch size)
        sample_z = np.random.uniform(-1, 1, size=(1,100))

        # Generate a sample by runnning generator(z)
        gen_sample = sess.run(generator(z, reuse=True), feed_dict={z:sample_z})

        # Add each gen_sample ( which is an array still ) to the list!
        samples.append(gen_sample)

# show me a sample image after we "un-flatten it"
# samples[n].shape = the nth image in the sample list where < less epochs
plt.imshow(samples[3].reshape(28,28))

### Don't save!  this is only running for a few epochs to test...

''''
saver = tf.train.Saver(var_list = g_vars)
new_samples = []

with tf.Session() as sess:
    saver.restore(sess,'./models/500_epoch_model.ckpt')

    for x in range(5):
        sample_z = np.random.uniform(-1,1,size=(1,100))
        gen_sample = sess.run(generator(z,reuse=True),feed_dict={z:sample_z})

        new_samples.append(gen_sample)

plt.imshow(new_samples[0].reshape(28,28))
'''
