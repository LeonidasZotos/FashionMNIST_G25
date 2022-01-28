# %%
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.metrics import (plot_confusion_matrix, plot_det_curve,
                             plot_roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
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

"""
Used to visualise the images
"""


def visualize_image(features, res_path):
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(features[i].reshape(28, 28), cmap="gray")
    # plt.show()
    plt.savefig(f"{res_path}/dataset_image.png")


# %%

"""
Load the training and testing data from the fashion-mnist data set
"""


def load_data(main_path, subset=None):
    df_train = pd.read_csv(str(main_path / "fashion-mnist_train.csv"))
    df_test = pd.read_csv(str(main_path / "fashion-mnist_test.csv"))
    print(df_train.head(5))

    # If a subset has been defined and is valid
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


def flip(x):  # Horizontal flip
    # 28*28
    x = np.reshape(x, (28, 28))
    x = np.fliplr(x)
    x = x.flatten()  # Return to 1d array
    return x


def gaussian_filter(x):
    # sigma=1 seems to give best results
    return ndimage.gaussian_filter(x, sigma=1)


def maxf(x):
    return ndimage.maximum_filter(x, 2)


def convolveOutline(x):
    weights = [-1, 4, -1]  # Outlining
    try:
        x = ndimage.convolve(x, weights)
    except RuntimeError:
        x = x[:, :, None]
        x = ndimage.convolve(x, weights)
    return x


def convolveSharpen(x):
    weights = [0, 2, 0]  # Sharpening
    try:
        x = ndimage.convolve(x, weights)
    except RuntimeError:
        x = x[:, :, None]
        x = ndimage.convolve(x, weights)
    return x


def visualize_transforms(features, res_path):
    from mpl_toolkits.axes_grid1 import ImageGrid

    im1 = features[0]
    im2 = flip(im1).reshape(28, 28)
    im3 = convolveOutline(im1).reshape(28, 28)
    im4 = convolveSharpen(im1).reshape(28, 28)
    im5 = gaussian_filter(im1).reshape(28, 28)
    im6 = maxf(im1).reshape(28, 28)

    _, axs = plt.subplots(3, 2)
    axs = axs.flatten()
    names = ["original", "flip", "outline", "sharpen", "gaussian", "maximum"]
    i = 0
    for img, ax in zip([im1.reshape(28, 28), im2, im3, im4, im5, im6], axs):
        ax.imshow(img, cmap="gray")
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title(names[i])
        i += 1

    plt.savefig(f"{res_path}/transform_image.png")


def return_process(im):
    list_of_transforms = [
        lambda x: x,
        flip,
        gaussian_filter,
        convolveOutline,
        convolveSharpen,
    ]  # first returns the original (no transform)
    # "weights" determines probability to choose each transformation
    chosenTransform = random.choices(list_of_transforms, weights=(0, 1, 1, 1, 1))
    return chosenTransform[0](im)


def preprocess_skeleton(array, labels, disable=False, sequential=True):
    if disable == True:
        return array, labels
    else:
        if sequential == True:
            # array = parallel(return_process, arr=array)
            percentToTransform = (
                0.5  # use 0.5 to transform 50% of the data and append to the end
            )
            originalLength = len(array)
            numberOfTransforms = math.floor(len(array) * percentToTransform)

            for i in range(0, numberOfTransforms):
                chosenIndex = random.randrange(originalLength)
                transformedImage = return_process(array[chosenIndex])
                # add transformed image to set
                array = np.vstack((array, transformedImage))
                # also copy the label
                labels = np.append(labels, labels[chosenIndex])

            return array, labels
        else:
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
    X, train_labels = preprocess_skeleton(
        X,
        train_labels,
    )  # enable for processing

    visualize_image(X, res_path)
    visualize_transforms(X, res_path)
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


"""
Method used to do validation of the models in the model list over a predefined list of parameters
"""


def validation_stage(
    model_list,
    model_parameters,
    train_features,
    train_labels,
    test_features,
    test_labels,
):
    # Perform the Validation using Sklearn GridSearch

    model_best_params = []

    print("[INFO] Performing Validation")
    for i in range(len(tqdm(model_list))):
        print("[VALIDATION] %s model being validated\n" % (model_list[i]))
        grid = GridSearchCV(
            model_list[i](), model_parameters[i], verbose=3, refit=True, n_jobs=12
        )
        grid.fit(train_features, train_labels)

        model_best_params.append(grid.best_params_)

        grid_predictions = grid.predict(test_features)

        print(classification_report(test_labels, grid_predictions))
        print(
            "[VALIDATION] %s model validation completed\n"
            "==========================================\n" % (model_list[i])
        )

    print("[INFO] Validation Stage Completed")

    return model_best_params


"""
Method used to run multiple models for comparison
"""


def multi_model_run(
    train_features,
    test_features,
    train_labels,
    test_labels,
    model_list,
    model_parameters,
    reduce_dims,
    metrics,
    res_path,
    val_features,
    val_labels,
    folds=10,
):
    # Determine the best parameters for each of the models defined in model list and from the list of model parameters
    model_best_params = validation_stage(
        model_list,
        model_parameters,
        train_features,
        train_labels,
        test_features,
        test_labels,
    )

    # Create instances of the models in the model lists while making use of the best
    #   parameters attained in the validation stage
    print("[INFO] Applying Best Parameters to Models")
    for i in range(len(tqdm(model_list))):
        model_list[i] = model_list[i](**model_best_params[i])

    print("[INFO] Training and Testing Optimised Models")

    final_dict_results = {}
    results = Parallel(n_jobs=2)(
        delayed(train_and_predict)(
            train_features,
            test_features,
            train_labels,
            test_labels,
            val_features,
            val_labels,
            m,
            metrics,
            res_path,
            reduce_dims,
            folds,
        )
        for m in tqdm(model_list)
    )
    print("[INFO] Models Done Running\n[INFO] Compiling Results")

    final_dict_results = {
        str(model_list[i]): results[i] for i in range(len(model_list))
    }

    # print(final_dict_results)
    df = pd.DataFrame.from_dict(final_dict_results)
    df.to_csv(f"{res_path}/outputs.csv", mode="a")
    return final_dict_results


# %%
