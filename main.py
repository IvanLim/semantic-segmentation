import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Load the model and get the default graph    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)    
    graph = tf.get_default_graph()
    
    # From the default graph, get the placeholder variables
    tf_input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    tf_keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

    # and the layers
    tf_vgg_frozen_layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    tf_vgg_frozen_layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    tf_vgg_frozen_layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return tf_input_image, tf_keep_prob, tf_vgg_frozen_layer3, tf_vgg_frozen_layer4, tf_vgg_frozen_layer7
tests.test_load_vgg(load_vgg, tf)


def layers(tf_vgg_frozen_layer3, tf_vgg_frozen_layer4, tf_vgg_frozen_layer7, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # Perform a 1x1 convolutions on the vgg layers to preserve spatial information
    pool3 = tf.layers.conv2d(tf_vgg_frozen_layer3, num_classes, 1, strides=(1, 1), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    pool4 = tf.layers.conv2d(tf_vgg_frozen_layer4, num_classes, 1, strides=(1, 1), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    conv7 = tf.layers.conv2d(tf_vgg_frozen_layer7, num_classes, 1, strides=(1, 1), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Perform upsampling, and add skip layers, as per 
    # the FCN-8 architecture https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
    # which is:
    # 8x_upsample( 4x_upsample(conv7) + 2x_upsample(pool4) + pool3 )
    #
    # Convolutions are commutative, so there are many ways to skin this cat. We'll go with:
    #
    #   8x_upsample( 2x_upsample(2x_upsample(conv7) + pool4) + pool3 )

    # 2x_upsample(conv7)
    output = tf.contrib.layers.conv2d_transpose(conv7, num_classes, 4, strides=(2, 2), padding='same', weights_regularizer=tf.contrib.layers.l2_regularizer(1e-3), weights_initializer=tf.contrib.layers.xavier_initializer())

    # 2x_upsample(conv7) + pool4
    output = tf.add(output, pool4)

    # 2x_upsample(2x_upsample(conv7) + pool4)
    output = tf.contrib.layers.conv2d_transpose(output, num_classes, 4, strides=(2, 2), padding='same', weights_regularizer=tf.contrib.layers.l2_regularizer(1e-3), weights_initializer=tf.contrib.layers.xavier_initializer())

    # 2x_upsample(2x_upsample(conv7) + pool4) + pool3
    output = tf.contrib.add(output, pool3)

    # 8x_upsample( 2x_upsample(2x_upsample(conv7) + pool4) + pool3 )
    output = tf.contrib.layers.conv2d_transpose(output, num_classes, 16, strides=(8, 8), padding='same', weights_regularizer=tf.contrib.layers.l2_regularizer(1e-3), weights_initializer=tf.contrib.layers.xavier_initializer())

    return output
tests.test_layers(layers)


def optimize(nn_last_layer, tf_correct_label, tf_learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    
    # Reshape the last layer from a 4D tensor to a 2D tensor
    # Each row represents a pixel, and each column represents a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    # Loss function that we want to minimize
    tf_cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_correct_label))

    # Define our optimizer, which tries to minize the cross entropy loss at a given learning rate
    tf_train_op = tf.train.AdamOptimizer(tf_learning_rate).minimize(tf_cross_entropy_loss)

    return logits, tf_train_op, tf_cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param loss_op: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    print ("Training...")

    # For each training epoch
    for epoch in range(1, epochs + 1):
        
        print(" Epoch: ", epoch)

        batch_num = 0

        # For each image-label pair in batch
        for image, label in get_batches_fn(batch_size):
            
            # We use a smaller learning rate (1e-4) because we have a small batch size
            # which might cause the gradients to be unstable
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, 
                                                               correct_label: label, 
                                                               keep_prob: 0.5, 
                                                               learning_rate: 1e-4})
            batch_num += 1
            print ("  - Batch: {} Loss: {}".format(batch_num, loss))

    print ("Training completed")

tests.test_train_nn(train_nn)


def run():
    epochs = 20
    batch_size = 8

    num_classes = 2
    image_shape = (160, 576)

    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Load tensors / frozen layers from the vgg16 model
        tf_input_image, tf_keep_prob, tf_vgg_frozen_layer3, tf_vgg_frozen_layer4, tf_vgg_frozen_layer7 = load_vgg(sess, vgg_path)

        # Add our deconvolution layers and skip layers to make it a full FCN
        layer_output = layers(tf_vgg_frozen_layer3, tf_vgg_frozen_layer4, tf_vgg_frozen_layer7, num_classes)

        # Set up the two remaining tf_placeholders to be plugged into our tensorflow functions
        # Logits and labels must have the same shape. Our logits are 4D tensors so labels will need to match
        tf_correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
        tf_learning_rate = tf.placeholder(tf.float32)

        logits, train_op, cross_entropy_loss = optimize(layer_output, tf_correct_label, tf_learning_rate, num_classes)

        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, tf_input_image, tf_correct_label, tf_keep_prob, tf_learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, tf_keep_prob, tf_input_image)

if __name__ == '__main__':
    run()
