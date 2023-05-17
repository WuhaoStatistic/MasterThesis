import numpy as np
import os

from tensorflow.keras.layers import Conv2D, Dropout
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from layers_solved import *

CLASS_WEIGHT = [0.27163427, 10.66462364, 6.64531657, 13.45360825]


class UNet(tf.keras.Model):
    """The UNet model."""

    def __init__(self, n_classes, class_weights=None, filters_start=64, kernel_size=3, depth=3, drop_rate=0.2,
                 name=None):
        super(UNet, self).__init__(name=name)

        if class_weights is None:
            class_weights = CLASS_WEIGHT
        self.depth = depth
        self.n_classes = n_classes
        self.class_weights = class_weights

        ########## YOUR CODE ############
        self.encoder = [EncoderLayer(filters_start * (2 ** i), kernel_size) for i in range(depth)]

        self.drop1 = Dropout(drop_rate)
        self.bridge = ConvsLayer(filters_start * (2 ** depth), kernel_size, name="Bridge")
        self.drop2 = Dropout(drop_rate)

        self.decoder = [DecoderLayer(filters_start * (2 ** i), kernel_size) for i in range(depth - 1, -1, -1)]
        self.classifier = Conv2D(
            filters=n_classes,
            kernel_size=1,
            padding="same",
            kernel_initializer="he_normal",
            activation="softmax",
            name="Classifier",
        )
        ##########    END    ############

    def call(self, inp):

        ########## YOUR CODE ############

        lays = []
        x = inp

        # Encoding
        for encoder_layer in self.encoder:
            lay, x = encoder_layer(x)
            lays.append(lay)

        # Bottleneck
        x = self.drop1(x)
        x = self.bridge(x)
        x = self.drop2(x)

        # Decoding
        for decoder_layer in self.decoder:
            x = decoder_layer(x, lays.pop())

        # Classifing
        out = self.classifier(x)

        return out

        ##########    END    ############

    def dice(self, y_true, y_pred, weights=None):
        """Calculate the mean dice score"""

        if weights is None:
            weights = self.class_weights

        weights = tf.convert_to_tensor(weights[None, None, None, :], tf.float32)
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        ########## YOUR CODE ############

        numerator = 2 * tf.math.reduce_sum(y_true * y_pred * weights)
        denominator = tf.math.reduce_sum((y_true + y_pred) * weights) + 1e-5

        dice_score = numerator / denominator

        ########## YOUR CODE ############

        return dice_score

    def dice_loss(self, y_true, y_pred, weights=None):
        """Calculate the mean dice score"""

        return 1 - self.dice(y_true, y_pred, weights)

    def train(self, train_gen, valid_gen, num_epochs, steps_per_epoch, optimizer=None):
        """Train the model"""

        if optimizer is None:
            optimizer = Adam(learning_rate=1e-4)

        self.compile(optimizer=optimizer, loss=self.dice_loss, metrics=["acc", self.dice])

        self.ckpt_path = "./train/" + self.name + "/"

        callbacks = [
            ModelCheckpoint(self.ckpt_path, save_best_only=True, save_weights_only=True, verbose=1),
            EarlyStopping(patience=5),
        ]

        history = self.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
        )

        return history

    def calculate_metrics(self, y_true_flat, y_pred_flat):
        """Calculate accuracy and Dice score between two binary vectors"""

        ########## YOUR CODE ############
        # Calculate the confusion matrix using tf.math.confusion_matrix()
        cm = tf.math.confusion_matrix(y_true_flat, y_pred_flat, num_classes=2)

        # Calculate the accuracy and Dice using the confusion matrix
        # Set to np.nan the value for Dice when the confusion matrix has only true negatives.

        acc = tf.linalg.trace(cm) / tf.reduce_sum(cm)

        if cm[0, 0] == len(y_true_flat):
            dice = np.nan
        else:
            dice = 2 * cm[1, 1] / (2 * cm[1, 1] + cm[1, 0] + cm[0, 1])

        ##########    END    ############

        return acc, dice

    def get_metrics(self, dataset):
        """Calculates the metrics accuracy and Dice over the input generator."""

        Nim = len(list(dataset.as_numpy_iterator()))
        ACC = np.empty((Nim, self.n_classes))
        DICE = np.empty((Nim, self.n_classes))
        n = 0
        for X, y_true in dataset:
            y_pred = self(X)
            y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=self.n_classes, off_value=0.0, dtype=tf.float32)

            for c in range(self.n_classes):
                y_true_flat = tf.reshape(y_true[0, :, :, c], [-1])
                y_pred_flat = tf.reshape(y_pred[0, :, :, c], [-1])

                acc, dice = self.calculate_metrics(y_true_flat, y_pred_flat)
                ACC[n, c] = acc
                DICE[n, c] = dice

            n += 1

        return ACC, DICE
