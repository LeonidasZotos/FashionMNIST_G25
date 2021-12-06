# %%
import os
from operator import mod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from datetime import datetime

# %%

label_array = {
    0: "top",
    1: "trouser",
    2: "pullover",
    3: "dress",
    4: "coat",
    5: "sandal",
    6: "shirt",
    7: "sneaker",
    8: "bag",
    9: "ankle_boot",
}

def return_shape(tes):
    print(tes.shape)


# %%


def print_shapes(arr):
    for i in arr:
        return_shape(i)


# %%


def visualize_image(features, res_path):
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(features[i].reshape(28, 28))
    # plt.show()
    plt.savefig(f"{res_path}/dataset_image.png")


# %%


def load_data(main_path, subset=None):
    df_train = pd.read_csv(str(main_path / "fashion-mnist_train.csv"))
    df_test = pd.read_csv(str(main_path / "fashion-mnist_test.csv"))
    print(df_train.head(5))

    if subset != None and subset > 0:
        df_train = df_train.head(subset)
        df_test = df_test.head(subset)

    train_features, val_features, train_labels, val_labels = train_test_split(
        df_train.drop("label", axis=1), df_train["label"], test_size=0.2
    )
    test_features, test_labels = df_test.drop("label", axis=1), df_test["label"]
    (
        train_features,
        val_features,
        train_labels,
        val_labels,
        test_features,
        test_labels,
    ) = (
        train_features.to_numpy(),
        val_features.to_numpy(),
        train_labels.to_numpy(),
        val_labels.to_numpy(),
        test_features.to_numpy(),
        test_labels.to_numpy(),
    )
    print("[INFO] DONE LOADING DATA")
    return (
        train_features,
        val_features,
        train_labels,
        val_labels,
        test_features,
        test_labels,
    )


# %%
# ML PIPELINE


def preproces_skeleton(array, process=np.flip):
    # currently does for all images in array
    # TODO : Add results to the array
    if process == None:
        return array
    else:
        for i in range(array.shape[0]):
            array[i] = process(array[i])
        return array


def dimensionality_reduction(X, X_test, method="pca"):
    if method == "pca":
        pca = PCA(0.85)
        pca.fit(X)
        X = pca.transform(X)
        X_test = pca.transform(X_test)


def train_and_predict(
    train_features,
    test_features,
    train_labels,
    test_labels,
    model,
    metrics,
    res_path,
    reduce_dims=None,
):
    # scale data
    X = StandardScaler().fit_transform(train_features)
    X_test = StandardScaler().fit_transform(test_features)

    # preprocess_step
    X = preproces_skeleton(X, None)
    if reduce_dims != None:
        dimensionality_reduction(X, X_test, reduce_dims)

    # fit model
    model.fit(X, train_labels)

    y1 = test_labels
    y2 = model.predict(X_test)

    # metrics
    if type(metrics) != list:
        metrics = [metrics]

    dict_results = {}
    for metric in metrics:
        try:
            dict_results[metric.__name__] = metric(y1, y2)
        except ValueError:
            dict_results[metric.__name__] = metric(y1, y2, average="macro")

    plt.cla()
    plt.clf()
    plot_confusion_matrix(model, X_test, test_labels)
    plt.savefig(f"{res_path}/confusion_{str(model)}.png")

    print(dict_results)
    return dict_results


def multi_model_run(
    train_features,
    test_features,
    train_labels,
    test_labels,
    model_list,
    reduce_dims,
    metrics,
    res_path
):
    final_dict_results = {}
    for model in tqdm(model_list):
        final_dict_results[str(model)] = train_and_predict(
            train_features=train_features,
            test_features=test_features,
            train_labels=train_labels,
            test_labels=test_labels,
            model=model,
            reduce_dims=reduce_dims,
            metrics=metrics,
            res_path=res_path
        )
    print(final_dict_results)
    df = pd.DataFrame.from_dict(final_dict_results)
    df.to_csv(f"{res_path}/outputs.csv", mode="a")
    return final_dict_results
