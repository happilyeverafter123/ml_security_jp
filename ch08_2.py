# import numpy as np
# import tensorflow as tf
# import os
# from art.estimators.classification.tensorflow import TensorFlowV2Classifier
# from art.utils import load_mnist

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# (X_train, y_train), (X_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# nb_classes = 10

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Conv2D(
#     1,
#     kernel_size=(7, 7),
#     activation='relu',
#     input_shape=(28, 28, 1)
# ))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(nb_classes, activation='relu', input_shape=(28, 28, 1)))
# model.compile(metrics=['accuracy'])

# victim_classifier = TensorFlowV2Classifier(model,
#                     nb_classes=nb_classes,
#                     input_shape=(28, 28, 1),
#                     loss_object=tf.keras.losses.CategoricalCrossentropy(),
#                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
#                                         )

# victim_classifier.fit(X_train, y_train, nb_epochs=5, batch_size=128)

# preds = victim_classifier.predict(x=X_test[:100])
# acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test[:100], axis=1)) / len(y_test)
# print("Accuracy of the victim model on the test set: {}".format(acc * 100))

# ZOO Attack

import lightgbm as lgb

from art.estimators.classification import LightGBMClassifier
from art.utils import load_mnist

(X_train, y_train), (X_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

X_test = X_test[0:5]
y_test = y_test[0:5]

nb_sample_train = X_train.shape[0]
nb_sample_test = X_test.shape[0]
X_train = X_train.reshape((nb_sample_train, 28 * 28))
          
