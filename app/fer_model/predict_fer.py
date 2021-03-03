import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import dlib
import cv2


def predict_facial_expression(image):
    image_size = 128
    num_channels = 3
    images = []
    classes = ["happy", "sad", "angry"]

    # Prepare image to input into model
    image = cv2.imread(image)
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype("float32")
    images = np.multiply(images, 1.0 / 255.0)

    # Reshape image to the same shape as the layers
    x_batch = images.reshape(1, image_size, image_size, num_channels)

    # Restore session to recreate TensorFlow graph of FER model
    sess = tf.Session()
    saver = tf.train.import_meta_graph("app/fer_model/fer-model.meta")
    saver.restore(sess, tf.train.latest_checkpoint("app/fer_model/"))
    graph = tf.get_default_graph()

    # Retrieve predictions from model
    y_pred = graph.get_tensor_by_name("y_pred:0")

    # Get placeholder variables
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 3))

    # Creating the same dictionary that is used in training the model
    feed_dict_testing = {x: x_batch, y_true: y_test_images}

    # Get prediction
    prediction = sess.run(y_pred, feed_dict=feed_dict_testing)
    prediction = prediction.tolist()
    print(prediction)
    for index, p in enumerate(prediction[0]):
        prediction[0][index] = float("{0:.4f}".format(p))

    print(prediction)
    value = max(prediction[0])
    index = prediction[0].index(value)
    emotion = classes[index]
    return str(emotion), str(value)


def detect_face(image):
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(image, 0)
    dedected = detector(img, 1)
    if len(dedected) == 0:
        raise Exception("No face detected, please try again.")
    elif len(dedected) > 1:
        raise Exception("More than one face detected, please try again.")
    else:
        for i, d in enumerate(dedected):
            crop = img[d.top() : d.bottom(), d.left() : d.right()]
            cv2.imwrite(image, crop)
