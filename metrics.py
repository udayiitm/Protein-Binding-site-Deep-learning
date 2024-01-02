import tensorflow as tf
from tensorflow.keras import backend as K


def iou_metric(y_true, y_pred):
    # Reshape the inputs to binary 1D arrays
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    y_pred = tf.cast(y_pred >= 0.5, tf.float32)

    # Calculate true positives, false positives, and false negatives
    true_positives = tf.reduce_sum(y_true * y_pred)
    false_positives = tf.reduce_sum((1 - y_true) * y_pred)
    false_negatives = tf.reduce_sum(y_true * (1 - y_pred))

    all_predicted_positives = true_positives + false_positives
    all_actual_positives = true_positives + false_negatives

    # Compute the intersection and union of the true and predicted labels
    intersection = true_positives
    union = all_actual_positives + all_predicted_positives - intersection

    # Compute the IOU score
    iou = (intersection + K.epsilon()) / (union + K.epsilon())

    return iou


def dice_loss(y_true, y_pred):
    # Reshape the inputs to binary 1D arrays
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    # Calculate true positives, false positives, and false negatives
    true_positives = tf.reduce_sum(y_true * y_pred)
    false_positives = tf.reduce_sum((1 - y_true) * y_pred)
    false_negatives = tf.reduce_sum(y_true * (1 - y_pred))

    all_predicted_positives = true_positives + false_positives
    all_actual_positives = true_positives + false_negatives

    # Compute the intersection and union of the true and predicted labels
    intersection = true_positives
    # union = all_actual_positives + all_predicted_positives - intersection

    dice = (2.0 * intersection + K.epsilon()) / (
        all_actual_positives + all_predicted_positives + K.epsilon()
    )

    return 1 - dice


def f1_score(y_true, y_pred):
    """
    Calculates F1 score for binary classification.
    """
    # Reshape the inputs to binary 1D arrays
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.round(tf.reshape(y_pred, [-1]))

    # Calculate true positives, false positives, and false negatives
    true_positives = tf.reduce_sum(y_true * y_pred)
    false_positives = tf.reduce_sum((1 - y_true) * y_pred)
    false_negatives = tf.reduce_sum(y_true * (1 - y_pred))

    all_predicted_positives = true_positives + false_positives
    all_actual_positives = true_positives + false_negatives

    precision = true_positives / (all_predicted_positives + K.epsilon())
    recall = true_positives / (all_actual_positives + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return f1


def f2_score(y_true, y_pred, beta=2):
    """
    Calculates F2 score for binary classification.
    """
    # Reshape the inputs to binary 1D arrays
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.round(tf.reshape(y_pred, [-1]))

    # Calculate true positives, false positives, and false negatives
    true_positives = tf.reduce_sum(y_true * y_pred)
    false_positives = tf.reduce_sum((1 - y_true) * y_pred)
    false_negatives = tf.reduce_sum(y_true * (1 - y_pred))

    all_predicted_positives = true_positives + false_positives
    all_actual_positives = true_positives + false_negatives

    precision = true_positives / (all_predicted_positives + K.epsilon())
    recall = true_positives / (all_actual_positives + K.epsilon())

    f2 = (
        (1 + beta**2)
        * (precision * recall)
        / ((beta**2 * precision) + recall + K.epsilon())
    )

    return f2


def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    """
    Tversky loss function for 3D image segmentation model.

    :param alpha: Weight of false positives (float)
    :param beta: Weight of false negatives (float)

    :return: Tversky loss (float)
    """

    # Reshape the inputs to binary 1D arrays
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    # Calculate true positives, false positives, and false negatives
    true_positives = tf.reduce_sum(y_true * y_pred)
    false_positives = tf.reduce_sum((1 - y_true) * y_pred)
    false_negatives = tf.reduce_sum(y_true * (1 - y_pred))

    # Calculate Tversky index
    tversky_index = (true_positives + K.epsilon()) / (
        true_positives + alpha * false_positives + beta * false_negatives + K.epsilon()
    )

    # Calculate Tversky loss
    return 1 - tversky_index


# ====================================================
# ====================================================
# ====================================================


# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# # Define a simple Keras model that takes an input tensor and applies a dense layer
# inputs = keras.Input(shape=(36, 36, 36, 18), dtype=tf.float32)
# outputs = layers.Convolution3D(
#     filters=1,
#     kernel_size=1,
#     activation="sigmoid",
# )(inputs)
# model = keras.Model(inputs=inputs, outputs=outputs)

# # Compile the model with the custom metric
# model.compile(
#     optimizer="Adam",
#     loss=tversky_loss,
#     metrics=[
#         f1_score,
#         f2_score,
#         dvo_metric,
#         dice_loss,
#     ],
# )

# # Generate some random input and label data
# import numpy as np

# np.random.seed(123)
# x = np.random.rand(100, 36, 36, 36, 18)

# y = np.random.randint(0, 2, size=(100, 36, 36, 36))

# # Train the model and print the F1 score for each epoch
# history = model.fit(
#     x,
#     y,
#     epochs=2,
#     batch_size=32,
#     validation_split=0.2,
#     verbose=1,
# )

# # Evaluate the model on a test set and print the F1 score
# x_test = np.random.rand(1, 36, 36, 36, 18)
# y_test = np.random.randint(0, 2, size=(1, 36, 36, 36))
# score = model.evaluate(x_test, y_test, verbose=0)
# print("Test Traversky Loss:", score[0])
# print("Test F1 score:", score[1])
# print("Test F2 score:", score[2])
# print("Test DVO metric score:", score[3])
# print("Test Dice Loss score:", score[4])

# import matplotlib.pyplot as plt

# # Plot the training and validation loss and accuracy
# plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.plot(history.history["f1_score"], label="f1_score")
# plt.plot(history.history["f2_score"], label="f2_score")
# plt.plot(history.history["dvo_metric"], label="dvo_metric")
# plt.plot(history.history["dice_loss"], label="dice_loss")
# plt.legend()
# plt.xlabel("Epoch")
# plt.ylabel("Loss/Accuracy")
# plt.title("Training History")
# plt.savefig(fname="training_history")
# plt.show()
