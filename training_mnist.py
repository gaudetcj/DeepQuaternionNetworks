import numpy as np
import keras
from keras.models import Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Activation, Convolution2D, concatenate
from quaternion_layers.dense import QuaternionDense
from quaternion_layers.conv import QuaternionConv2D
from quaternion_layers.bn import QuaternionBatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 1200

def learnVectorBlock(I):
    """Learn initial vector component for input."""

    O = Convolution2D(1, (5, 5),
                      padding='same', activation='relu')(I)

    return O

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    width_shift_range=0.1,
    height_shift_range=0.1)

datagen.fit(x_train)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

R = Input(shape=input_shape)
I = learnVectorBlock(R)
J = learnVectorBlock(R)
K = learnVectorBlock(R)
O = concatenate([R, I, J, K], axis=-1)
O = QuaternionConv2D(64, (5, 5), activation='relu', padding="same", kernel_initializer='quaternion')(O)
O = QuaternionConv2D(64, (5, 5), activation='relu', padding="same", kernel_initializer='quaternion')(O)
O = QuaternionConv2D(32, (5, 5), activation='relu', padding="same", kernel_initializer='quaternion')(O)

# O = Convolution2D(256, (5, 5), activation='relu', padding="same")(R)
# O = Convolution2D(256, (5, 5), activation='relu', padding="same")(O)
# O = Convolution2D(128, (5, 5), activation='relu', padding="same")(O)

O = Flatten()(O)
O = QuaternionDense(82, activation='relu', kernel_initializer='quaternion')(O)
O = QuaternionDense(48, activation='relu', kernel_initializer='quaternion')(O)
#O = Dropout(0.5)(O)
O = Dense(num_classes, activation='softmax')(O)

model = Model(R, O)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                           steps_per_epoch=len(x_train) / batch_size, epochs=epochs,
                           validation_data=(x_test, y_test))

np.save('mnist_results.npy', hist.history['val_acc'])