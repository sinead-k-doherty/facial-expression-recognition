import dataset
import tensorflow.compat.v1 as tf
from numpy.random import seed

tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()

total_iterations = 0


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(
    input, num_input_channels, conv_filter_size, num_filters
):
    # Creating filters
    weights = create_weights(
        shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters]
    )
    # Creating biase for each filter
    biases = create_biases(num_filters)
    # Strides sets how the filters move across the image e.g. 1 pixel at a time
    layer = tf.nn.conv2d(
        input=input, filter=weights, strides=[1, 1, 1, 1], padding="SAME"
    )
    layer += biases
    layer = create_max_pooling_layer(layer)
    return tf.nn.relu(layer)


def create_max_pooling_layer(layer):
    # 2x2 max pooling filter selects the largest pixel value in the filter
    # Halves the size of the image resolution
    return tf.nn.max_pool(
        value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 3, 1], padding="SAME"
    )


# Regular pooling layer to flatten previous layer for the next fully connected layer
def create_pooling_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    return tf.reshape(layer, [-1, num_features])


def create_fully_connected_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


def show_progress(
    epoch, feed_dict_train, feed_dict_validate, val_loss, session, accuracy
):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} - Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


def train(
    num_iteration,
    session,
    accuracy,
    data,
    batch_size,
    x,
    y_true,
    optimiser,
    cost,
    saver,
):
    global total_iterations

    for i in range(total_iterations, total_iterations + num_iteration):
        # Get a batch of training and validation images and labels
        x_batch, y_true_batch, _, _ = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, _ = data.valid.next_batch(batch_size)

        # Use the placeholder variables to create dictionaries of training and validation images and labels
        feed_dict_tr = {x: x_batch, y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

        # Run the optimiser on this batch of data
        session.run(optimiser, feed_dict=feed_dict_tr)

        # Print progress after total number of training images/33
        if i % int(data.train.num_examples / batch_size) == 0:
            # Calculate loss cost
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / batch_size))
            show_progress(
                epoch, feed_dict_tr, feed_dict_val, val_loss, session, accuracy
            )
            # Save model
            saver.save(session, "./app/fer_model/fer-model2")

    total_iterations += num_iteration


def build_cnn_model():
    # Setting seed for random initialisation
    # For stable results when training
    seed(1)
    tf.set_random_seed(2)

    # Train in the images in batches of 33 images
    batch_size = 33

    classes = ["happy", "sad", "angry"]
    num_classes = len(classes)
    img_size = 128
    num_channels = 3

    # Get the dataset
    data = dataset.read_train_sets(img_size, classes, data_type="images")

    print("Number of training images: {}".format(len(data.train.labels)))
    print("Number of validation images: {}".format(len(data.valid.labels)))

    # Create TensorFlow session to execute TensorFlow graph
    session = tf.Session()

    # Placeholder variable for storing input images
    x = tf.placeholder(
        tf.float32, shape=[None, img_size, img_size, num_channels], name="x"
    )

    # Corresponding labels for images
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name="y_true")

    # Placeholder variable to calculate the number of classes
    y_true_cls = tf.argmax(y_true, dimension=1)

    # Setting number of filters and filter sizes for all layers
    filter_size_conv1 = 3
    num_filters_conv1 = 32

    filter_size_conv2 = 3
    num_filters_conv2 = 32

    filter_size_conv3 = 3
    num_filters_conv3 = 64

    fc_layer_size = 128

    # Creating layers for model
    layer_conv1 = create_convolutional_layer(
        input=x,
        num_input_channels=num_channels,
        conv_filter_size=filter_size_conv1,
        num_filters=num_filters_conv1,
    )

    layer_conv2 = create_convolutional_layer(
        input=layer_conv1,
        num_input_channels=num_filters_conv1,
        conv_filter_size=filter_size_conv2,
        num_filters=num_filters_conv2,
    )

    layer_conv3 = create_convolutional_layer(
        input=layer_conv2,
        num_input_channels=num_filters_conv2,
        conv_filter_size=filter_size_conv3,
        num_filters=num_filters_conv3,
    )

    layer_pooling = create_pooling_layer(layer_conv3)

    layer_fc1 = create_fully_connected_layer(
        input=layer_pooling,
        num_inputs=layer_pooling.get_shape()[1:4].num_elements(),
        num_outputs=fc_layer_size,
        use_relu=True,
    )

    layer_fc2 = create_fully_connected_layer(
        input=layer_fc1,
        num_inputs=fc_layer_size,
        num_outputs=num_classes,
        use_relu=False,
    )

    # Softmax function used my layer_fc2 to normalise prediction values for 3 classes
    y_pred = tf.nn.softmax(layer_fc2, name="y_pred")

    # Stores the index of the largest predicted value
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    session.run(tf.global_variables_initializer())

    # Loss function calculates how well the model is performing
    # Is used to reduce the error value of predictions by adjusting weights and biases
    loss_function = tf.nn.softmax_cross_entropy_with_logits(
        logits=layer_fc2, labels=y_true
    )
    cost = tf.reduce_mean(loss_function)
    # Used to optimise the model to reduce the error rate
    optimiser = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    # Calculate the number of times the model accurately predicts images
    correct_predictions = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    session.run(tf.global_variables_initializer())

    # Save training model
    saver = tf.train.Saver()

    # train model
    train(3000, session, accuracy, data, batch_size, x, y_true, optimiser, cost, saver)


build_cnn_model()
