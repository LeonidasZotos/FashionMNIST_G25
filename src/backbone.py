# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# %%
main_path = Path("../data/")
# %%
def return_shape(tes):
    print(tes.shape)


# %%


def print_shapes(arr):
    for i in arr:
        return_shape(i)


# %%


def visualize_image(features):
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        # imshow  takes an array ( with dimension = 2, RGB or B/W) and gives you the image that corresponds to it
        plt.imshow(features[i].reshape(28, 28))
        plt.show()


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


def train_and_predict(train_features, test_features, model, metrics, normalize=True):
    # scale data
    X = StandardScaler().fit_transform(train_features)
    X_test = StandardScaler().fit_transform(test_features)

    # preprocess_step
    X = preproces_skeleton(X, None)

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

    print(dict_results)
    return dict_results


def multi_model_run(train_features, test_features, model_list, metrics):
    final_dict_results = {}
    for model in model_list:
        final_dict_results[str(model)] = train_and_predict(
            train_features=train_features,
            test_features=test_features,
            model=model,
            metrics=metrics,
        )
    print(final_dict_results)
    return final_dict_results


# %%
# READ DATA : Dont forget to remove subset (set to None for full data)
(
    train_features,
    val_features,
    train_labels,
    val_labels,
    test_features,
    test_labels,
) = load_data(main_path=main_path, subset=5000)
print_shapes(
    [train_features, val_features, train_labels, val_labels, test_features, test_labels]
)

# %%
# SEE IF READING WORKED
visualize_image(train_features)

#%% Run entire pipeline
multi_model_run(
    train_features=train_features,
    test_features=test_features,
    model_list=[
        KNeighborsClassifier(n_neighbors=3),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    ],
    metrics=[
        accuracy_score,
        precision_score,
        recall_score,
    ],
)
# %%
