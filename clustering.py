import os
import numpy as np
from regex import T
import tensorflow as tf

from imageio import imread
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from os import listdir

from tomlkit import string


img_height = 180
img_width = 180
num_class = 4
test_size = 0.2


dataset_path = "D:\RecipePal\DATASETS"

# NOTE: Function below is the same as tf.keras.preprocessing.image_dataset_from_directory
# Refer to TensorFlow Core V2.8.0 Logs  https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
"""
train_ds = tf.keras.utils.image_dataset_from_directory(  # Used for training set
    dataset_path,  # Denotes file directory containg dataset
    # Denotes that 80% of our dataset will be utilized for training & 20% will be used for validation
    validation_split=0.2,
    subset="training",  # Denotes what set this is
    seed=123,  # random seed for shuffling & image transformations
    image_size=(img_height, img_width),  # Size to resize images
    batch_size=batch_size,  # Size of the batches of data
)

test_ds = tf.keras.utils.image_dataset_from_directory(  # Used for validation set
    dataset_path,  # Denotes file directory containg dataset
    # Denotes that 80% of our dataset will be utilized for training & 20% will be used for validation
    validation_split=0.2,
    subset="training",  # Denotes what set this is
    seed=123,  # random seed for shuffling & image transformations
    image_size=(img_height, img_width),  # Size to resize images
    batch_size=batch_size,  # Size of the batches of data
)

class_names = train_ds.class_names
"""


# Processing images into 1 Dimensional Arrays so we'll need to reshape all images.


def get_img(data_path):  # Function to obtain images and resize them.
    # Getting image array from path:
    img = imread(data_path)
    # 1 if grayscale_images else 3
    img = np.resize(img, (img_height, img_width))
    return img


def get_dataset(dataset_path):
    # Getting all data from data path:
    try:  # Check if dataset already exists, if it does load the data
        X = np.load("npy_dataset/X.npy")
        Y = np.load("npy_dataset/Y.npy")
    except:  # Otherwise create the data
        labels = listdir(dataset_path)  # Obtain labels from dataset files
        X = []
        Y = []
        # Iterate through the labeled file to obtain images related to that label
        # Utilize iteration as a label for the images, this will be our Y data.
        for i, label in enumerate(labels):
            # Obtain the file path for each image in relation to the label
            datas_path = dataset_path + "/" + label
            # File path to images in each labeled file
            for data in listdir(datas_path):
                img = get_img(datas_path + "/" + data)  # Obtain image
                X.append(img)  # Append image to X data
                Y.append(i)  # Append iteration integer to Y data as a label
        # Create dateset:
        X = 1 - np.array(X).astype("float32") / 255.0  # Normalize our data
        # Y = np.array(Y).astype('float32')  # Y data to array of type float
        # Categorize Y data into binary labels, refer to: https://www.geeksforgeeks.org/python-keras-keras-utils-to_categorical/
        # Y = to_categorical(Y, num_class)
        if not os.path.exists("npy_dataset/"):
            os.makedirs("npy_dataset/")
        np.save("npy_dataset/X.npy", X)  # Save Training Data
        np.save("npy_dataset/Y.npy", Y)  # Save Training Data
    (
        X,
        X_test,
        Y,
        Y_test,
    ) = train_test_split(  # Create training and test sets by splitting the array into random test/train subsets
        X, Y, test_size=test_size, random_state=42
    )  # refer to: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    np.save("npy_dataset/X_test.npy", X_test)  # Save Testing Data
    np.save("npy_dataset/Y_test.npy", Y_test)  # Save Testing Data
    return X, X_test, Y, Y_test


def get_info(
    cluster_labels, Y
):  # Relates best 'fitted' label with each cluster in the model, this returns a map of clusters with the key of a label
    ref_labels = {}
    for i in range(len(np.unique(kmeans.labels_))):
        index = np.where(cluster_labels == i, 1, 0)
        num = np.bincount(Y[index == 1]).argmax()  # THIS CAUSES AN ERROR?
        ref_labels[i] = num

    return ref_labels


get_dataset(dataset_path)


# Load entire dataset
X = np.load("npy_dataset\X.npy")
Y = np.load("npy_dataset\Y.npy")
X_test = np.load("npy_dataset\X_test.npy")
Y_test = np.load("npy_dataset\Y_test.npy")

X_test = X_test.astype("float32")


X = X.reshape(len(X), -1)
X_test = X_test.reshape(len(X_test), -1)


total_clusters = len(np.unique(Y_test))

kmeans = MiniBatchKMeans(n_clusters=total_clusters)

kmeans.fit(X)

kmeans.labels_

reference_labels = get_info(kmeans.labels_, Y)  # Get Ref Labels

# Obtain labels for ingredients percieved in the cluster
ingredient_labels = np.random.rand(len(kmeans.labels_))

for i in range(len(kmeans.labels_)):
    ingredient_labels[i] = reference_labels[kmeans.labels_[i]]

# Now we may compare the predicted labels vs the actual labels

print(ingredient_labels[400].astype("int"))
print(Y[:400])

print(accuracy_score(ingredient_labels, Y))
