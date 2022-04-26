import glob
import os
from black import out
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


from sklearn import metrics
from imageio import imread
from skimage.transform import resize
from keras_preprocessing.image import random_rotation
from sklearn import preprocessing
from keras.utils.all_utils import to_categorical
from keras.layers import BatchNormalization
from sklearn.ensemble import RandomForestClassifier
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from torch import le
from silence_tensorflow import silence_tensorflow


# Since we're not using a GPU we're going to get a bunch of messages in the terminal about it, this line removes that issue
silence_tensorflow()


class ConvNetRf():

    def __init__(self, dim=(256, 256)):
        self.dim = dim
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.label_encoder = {}

    def to_array_train(self):  # Obtain data a
        # DATASET DIRECTORY CAN CHANGE EASILY
        for directory_path in glob.glob("D:\RecipePal\DATASETS/*"):
            label = directory_path.split("\\")[-1]
            print(label)

            for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
                print(img_path)
                img = imread(img_path)
                img = resize(img, self.dim)
                img = random_rotation(img, rg=360)
                self.train_images.append(img)
                self.train_labels.append(label)

        self.train_images = np.array(self.train_images)
        self.train_labels = np.array(self.train_labels)

    def to_array_test(self):
        for directory_path in glob.glob("D:\RecipePal\DATASETS/*"):
            ing_label = directory_path.split("\\")[-1]
            print(ing_label)

            for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
                print(img_path)
                img = imread(img_path)
                img = resize(img, self.dim)
                img = random_rotation(img, rg=360)
                self.test_images.append(img)
                self.test_labels.append(ing_label)

        self.test_images = np.array(self.test_images)
        self.test_labels = np.array(self.test_labels)

    """
        def encodeLabels(self):
            self.label_encoder = preprocessing.LabelEncoder()
            self.label_encoder.fit(self.test_labels)
            test_labels_encoded = self.label_encoder.transform(self.test_labels)
            self.label_encoder.fit(self.train_labels)
            train_labels_encoded = self.label_encoder.transform(self.train_labels)
            X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = self.train_images, train_labels_encoded, self.test_images, test_labels_encoded
            return X_TRAIN, Y_TRAIN, X_TEST, Y_TEST

    """


cnn = ConvNetRf()


cnn.to_array_train()

cnn.to_array_test()

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(cnn.test_labels)
test_labels_encoded = label_encoder.transform(cnn.test_labels)
label_encoder.fit(cnn.train_labels)
train_labels_encoded = label_encoder.transform(cnn.train_labels)
x_train, y_train, x_test, y_test = cnn.train_images, train_labels_encoded, cnn.test_images, test_labels_encoded

one_hot_y_train = to_categorical(y_train)
one_hot_y_test = to_categorical(y_test)

activation = 'sigmoid'

feature_extractor = Sequential()
feature_extractor.add(Conv2D(32, 3, activation=activation,
                      padding='same', input_shape=(256, 256, 3)))
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


data_for_random_forest = feature_extractor.predict(x_train)

random_forest_model = RandomForestClassifier(n_estimators=50, random_state=42)

random_forest_model.fit(data_for_random_forest, y_train)

X_test_feature = feature_extractor.predict(x_test)

prediction_RF = random_forest_model.predict(X_test_feature)

prediction_RF = label_encoder.inverse_transform(prediction_RF)

joblib.dump(random_forest_model, './random_forest.joblib')

print("Accuracy For Random Forest Using CNN Features =",
      metrics.accuracy_score(cnn.test_labels, prediction_RF))

matrix = metrics.confusion_matrix(cnn.test_labels, prediction_RF)

sns.heatmap(matrix, annot=True)


img = x_test[1]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0)
input_img_features = feature_extractor.predict(input_img)
prediction_RF = random_forest_model.predict(input_img_features)[0]
prediction_RF = label_encoder.inverse_transform([prediction_RF])
print("The prediction for this image is: ", prediction_RF)
print("The actual label for this image is: ", cnn.test_labels[1])


# CNN with old dataset'
# ORB with old dataset


# CNN new dataset
# CNNRF with old dataset
