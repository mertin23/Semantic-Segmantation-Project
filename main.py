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
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    with tf.name_scope(vgg_tag):

        graph = tf.get_default_graph()
        vgg_input_tensor_name = 'image_input:0'
        vgg_keep_prob_tensor_name = 'keep_prob:0'
        vgg_layer3_out_tensor_name = 'layer3_out:0'
        vgg_layer4_out_tensor_name = 'layer4_out:0'
        vgg_layer7_out_tensor_name = 'layer7_out:0'
    
        input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
        tf.summary.image('input_image', input_image)
        keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        vgg_layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
        vgg_layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
        vgg_layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep_prob, vgg_layer3_out, vgg_layer4_out,\
        vgg_layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    vgg_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,
                              strides=(1, 1),
                              name='vgg_layer7')
    
    
    trans7 = tf.layers.conv2d_transpose(vgg_layer7, num_classes,
                                        4,
                                        strides=(2,2),
                                        padding='same',
                                        name='trans7')
    vgg_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,
                                      strides=(1, 1),
                                      name='vgg_layer4')
    skip_conn4 = tf.add(trans7, vgg_layer4, name='skip_connection4')
                                        
    trans4 = tf.layers.conv2d_transpose(skip_conn4, num_classes,
                                        4,
                                        strides=(2,2),
                                        padding='same',
                                        name='trans4')
                                        
    vgg_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,
                                  strides=(1, 1),
                                name='vgg_layer3')
                                        
    skip_conn3 = tf.add(trans4, vgg_layer3, name='skip_connection3')
                                        
    output = tf.layers.conv2d_transpose(skip_conn3, num_classes,
                                        16,
                                        strides=(8,8),
                                        padding='same',
                                        name='trans3')

    return output;

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    with tf.name_scope('logits'):
        logits = tf.reshape(nn_last_layer, (-1, num_classes))

    with tf.name_scope('correct_label'):
        cl_class0, cl_class1 = tf.split(correct_label, num_classes, axis=3)
 
    with tf.name_scope('cross_entropy_loss'):
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                    labels=correct_label))
        with tf.name_scope('total'):
            cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
 
    with tf.name_scope('train'):
        # tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, optimizer, cross_entropy_loss

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
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())
	
    print("Training...")
    print()
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        for image, label in get_batches_fn(batch_size):
            print(len(image))
            _, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict={input_image: image, correct_label: label,                                keep_prob: 0.8, learning_rate: 0.001})
            print("Loss: = {:.3f}".format(loss))
        print()

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config=config) as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out,
                               vgg_layer7_out, num_classes)

        learning_rate = tf.placeholder(tf.float32, name='learning-rate')

        correct_label = tf.placeholder(tf.float32,
                (None, image_shape[0], image_shape[1], num_classes),
                name='correct-label')

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer,
                                                        correct_label,
                                                        learning_rate,
                                                        num_classes)

        # TODO: Train NN using the train_nn function
        epochs = 50
        batch_size = 10
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
                 cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)


        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape,
                                      logits, keep_prob, input_image)


if __name__ == '__main__':
    run()
