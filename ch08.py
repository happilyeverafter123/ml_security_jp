import numpy as np
import tensorflow as tf
import os
from art.estimators.classification.tensorflow import TensorFlowV2Classifier
from art.utils import load_mnist

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

(X_train, y_train), (X_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

nb_classes = 10

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(
    1,
    kernel_size=(7, 7),
    activation='relu',
    input_shape=(28, 28, 1)
))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(nb_classes, activation='relu', input_shape=(28, 28, 1)))
model.compile(metrics=['accuracy'])

victim_classifier = TensorFlowV2Classifier(model,
                    nb_classes=nb_classes,
                    input_shape=(28, 28, 1),
                    loss_object=tf.keras.losses.CategoricalCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
                                        )

victim_classifier.fit(X_train, y_train, nb_epochs=5, batch_size=128)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Conv2D(
#     32,
#     kernel_size=(3, 3),
#     activation='relu',
#     input_shape=(28, 28, 1)
# ))

# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(nb_classes, activation='softmax'))
# model.compile(metrics=['accuracy'])
# thieved_classifier = TensorFlowV2Classifier(model,
#                     nb_classes=nb_classes,
#                     input_shape=(28, 28, 1),
#                     loss_object=tf.keras.losses.CategoricalCrossentropy(),
#                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
# )


# from art.attacks.extraction import CopycatCNN

# attack = CopycatCNN(
#     classifier=victim_classifier,
#     batch_size_fit=16,
#     batch_size_query=16,
#     nb_epochs=30,
#     nb_stolen=1000
#     )

# attack.extract(X_train, y_train, thieved_classifier=thieved_classifier)

# victim_preds = np.argmax(victim_classifier.predict(x=X_train[:100]), axis=1)
# thieved_preds = np.argmax(thieved_classifier.predict(x=X_train[:100]), axis=1)
# acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
# print("Accuracy of the thieved model on the victim model's training set: {}".format(acc * 100))

preds = victim_classifier.predict(x=X_test[:100])
acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test[:100], axis=1)) / len(y_test)
print("Accuracy of the victim model on the test set: {}".format(acc * 100))

# FGSM

# from art.attacks.evasion import FastGradientMethod

# eps = 0.9
# attack = FastGradientMethod(estimator=victim_classifier, eps=eps)
# X_test_adv = attack.generate(x=X_test[:100])
# preds = victim_classifier.predict(x=X_test_adv)
# acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test[:100], axis=1)) / len(y_test)
# print("Accuracy of the victim model on the adversarial test set: {}".format(acc * 100))

# from matplotlib import pyplot as plt
# plt.matshow(X_test_adv[0].reshape(28, 28), cmap='gray')

# from dotenv import load_dotenv

# load_dotenv()

# base_save_path = os.environ.get("SAVE_PATH")

# save_path = f"{base_save_path.rstrip('.png')}_FGSM_eps_{eps}.png"

# plt.savefig(save_path)

# Carlini & Wagner Attack

from art.attacks.evasion.carlini import CarliniL2Method

from art.utils import random_targets

attack = CarliniL2Method(classifier=victim_classifier,
                         targeted=True,
                         max_iter=10)
params = {'y': random_targets(y_test, victim_classifier.nb_classes)}

X_test_adv = attack.generate(x=X_test, **params)

from matplotlib import pyplot as plt
plt.matshow(X_test_adv[0].reshape(28, 28), cmap='gray')

from dotenv import load_dotenv

load_dotenv()

base_save_path = os.environ.get("SAVE_PATH")

save_path = f"{base_save_path.rstrip('.png')}_Carlini.png"

plt.savefig(save_path)
