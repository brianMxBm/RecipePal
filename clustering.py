import os
import numpy as np
from regex import T
from imageio import imread
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn import metrics
from os import listdir

img_height = 600
img_width = 600
dataset_path = "O:\Downloads\PARENT_TRAIN"  # Change This To The Dataset Path
test_size = 0.2


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
        # Create training and test sets by splitting the array into random test/train subsets
    ) = train_test_split(
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


def metric_data(model, data):
    print("Clusters: {}".format(model.n_clusters))
    print("Inertia: {}".format(model.inertia_))
    print("Homogeneity : {}".format(metrics.homogeneity_score(data, model.labels_)))


def accuracyTest():  # Determine Best
    cluster_number = [10, 16, 36, 64]
    for i in cluster_number:
        total_clusters = len(np.unique(Y_test))
        kmeans = MiniBatchKMeans(n_clusters=i)
        kmeans.fit(X)
        metric_data(kmeans, Y)
        kmeans.labels_
        reference_labels = get_info(kmeans.labels_, Y)  # Get Ref Labels
        ingredient_labels = np.random.rand(len(kmeans.labels_))

        for i in range(len(kmeans.labels_)):
            ingredient_labels[i] = reference_labels[kmeans.labels_[i]]

        print("Accuracy: {}".format(accuracy_score(ingredient_labels, Y)))
        print("\n")


get_dataset(dataset_path)

X = np.load("npy_dataset\X.npy")
Y = np.load("npy_dataset\Y.npy")
X_test = np.load("npy_dataset\X_test.npy")
Y_test = np.load("npy_dataset\Y_test.npy")

X = X.reshape(len(X), -1)
X_test = X_test.reshape(len(X_test), -1)

X_test = np.load("npy_dataset\X_test.npy")
Y_test = np.load("npy_dataset\Y_test.npy")


kmeans = MiniBatchKMeans(n_clusters=64)

kmeans.fit(X)

reference_labels = get_info(kmeans.labels_, Y)

ingredient_labels = np.random.rand(len(kmeans.labels_))


for i in range(len(kmeans.labels_)):

    ingredient_labels[i] = reference_labels[kmeans.labels_[i]]

print("Accuracy score : {}".format(accuracy_score(ingredient_labels, Y)))

print("\n")
