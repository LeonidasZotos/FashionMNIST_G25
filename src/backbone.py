# %%
import os
import random
from datetime import datetime
from operator import contains, mod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from .utils import *

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
        plt.imshow(features[i].reshape(28, 28), cmap="gray")
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


def return_process(im):
    list_of_procs = [np.flip]
    return random.choice(list_of_procs)(
        im
    )  # TODO :add probability and every other transform


def preproces_skeleton(array, labels):
    # TODO : Append results to the array instead of replacing it
    # TODO : Use the labels to make sure it is correct after you append something
    array = parallel(return_process, arr=array)
    return array, labels


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
    val_features,
    val_labels,
    model,
    metrics,
    res_path,
    reduce_dims=None,
    folds=10,
):
    # scale data
    train_features = np.concatenate(
        (train_features, test_features)
    )  # combining both here because k-fold will split them again into parts
    train_labels = np.concatenate((train_labels, test_labels))
    X = StandardScaler().fit_transform(train_features)
    X_test = StandardScaler().fit_transform(val_features)  # essentially validation set

    # preprocess_step
    X, train_labels = preproces_skeleton(X, train_labels)

    visualize_image(X, res_path)
    if reduce_dims != None:
        dimensionality_reduction(X, X_test, reduce_dims)

    # k -fold
    dict_results = {x.__name__: [] for x in metrics}

    for metric in metrics:
        if metric.__name__ in ["precision_score", "f1_score", "recall_score"]:
            dict_results[metric.__name__] = np.mean(
                cross_val_score(
                    model,
                    X,
                    train_labels,
                    scoring=make_scorer(metric, average="micro"),
                    cv=folds,
                    n_jobs=8,
                )
            )
        else:
            dict_results[metric.__name__] = np.mean(
                cross_val_score(
                    model,
                    X,
                    train_labels,
                    scoring=make_scorer(metric),
                    cv=folds,
                    n_jobs=8,
                )
            )

    print(dict_results)

    plt.cla()
    plt.clf()
    plot_confusion_matrix(model.fit(X, train_labels), X_test, val_labels)
    plt.savefig(f"{res_path}/confusion_{str(model)}.png")

    return dict_results


def multi_model_run(
    train_features,
    test_features,
    train_labels,
    test_labels,
    model_list,
    ##################
    model_parameters,
    ##################
    reduce_dims,
    metrics,
    res_path,
    val_features,
    val_labels,
    folds=10,
):
    num_params_per_model = []
    for params in model_parameters:
        num_params_per_model.append(len(params))

    ############################
    # TODO: VALIDATION: Figure out how to access list of models and edit params - Tumi
    ############################

    final_dict_results = {}
    for model in tqdm(model_list):
        final_dict_results[str(model)] = train_and_predict(
            train_features=train_features,
            test_features=test_features,
            train_labels=train_labels,
            test_labels=test_labels,
            val_features=val_features,
            val_labels=val_labels,
            model=model,
            reduce_dims=reduce_dims,
            metrics=metrics,
            res_path=res_path,
            folds=folds,
        )
    print(final_dict_results)
    df = pd.DataFrame.from_dict(final_dict_results)
    df.to_csv(f"{res_path}/outputs.csv", mode="a")
    return final_dict_results


# %%
