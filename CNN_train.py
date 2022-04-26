from random import Random
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns


from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from torch import le
from preprocess_cnn import DataGenerator
from keras_preprocessing import image


def training():
    # Preprocessed images for training purposes
    training_generator = DataGenerator('train')
    # Preprocessed images for validation purposes
    validation_generator = DataGenerator('test')

    print(training_generator.labels)

    activation = 'sigmoid'  # Try with ReLU

    feature_extractor = Sequential()

    feature_extractor.add(Conv2D(32, 3, activation=activation,
                                 padding='same', input_shape=(100, 100, 3)))

    feature_extractor.add(BatchNormalization())

    feature_extractor.add(Conv2D(32, 3, activation=activation,
                                 padding='same', kernel_initializer='he_uniform'))

    feature_extractor.add(BatchNormalization())

    feature_extractor.add(MaxPooling2D())

    feature_extractor.add(Conv2D(64, 3, activation=activation,
                                 padding='same', kernel_initializer='he_uniform'))

    feature_extractor.add(BatchNormalization())

    feature_extractor.add(Conv2D(64, 3, activation=activation,
                                 padding='same', kernel_initializer='he_uniform'))

    feature_extractor.add(BatchNormalization())

    feature_extractor.add(MaxPooling2D())

    feature_extractor.add(Flatten())

    x = feature_extractor.output
    x = Dense(128, activation=activation, kernel_initializer='he_uniform')(x)
    prediction_layer = Dense(4, activation='softmax')(x)

    cnn_model = Model(inputs=feature_extractor.input, outputs=prediction_layer)

    cnn_model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy', metrics=['accuracy'])

    history = cnn_model.fit(
        training_generator, validation_data=validation_generator, use_multiprocessing=True, epochs=1)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    """"
    prediction_generator = cnn_model.predict_generator(
        validation_generator, 224)

    print('got passed here')

    y_true = np.array([0]*112 + [1] * 112)
    y_pred = prediction_generator > 0.5

    matrix = confusion_matrix(y_true, y_pred)

    sns.heatmap(matrix, annot=True)


    img_width, img_height = 100, 100

    img = image.load_img('apple.jpg', target_size=(img_width, img_height))

    img - image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    prediction = np.argmax(cnn_model.predict(img))

    print(prediction)
    """
