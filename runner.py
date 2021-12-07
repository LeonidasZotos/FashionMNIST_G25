import sklearn.neural_network

from src.backbone import *

# %%
main_path = Path(os.getcwd()) / "data"
time = datetime.now().strftime('%d_%m_%Hh%M')
res_path = str(os.getcwd()) + "/outputs/" + time
# %%
# READ DATA : Dont forget to remove subset (set to None for full data)
(
    train_features,
    val_features,
    train_labels,
    val_labels,
    test_features,
    test_labels,
) = load_data(main_path=main_path, subset=1000)
print_shapes(
    [train_features, val_features, train_labels, val_labels, test_features, test_labels]
)

# %%
# SEE IF READING WORKED + make folder for results
if not Path.is_dir(Path(res_path)):
    os.mkdir(res_path)
visualize_image(train_features, res_path)

# %% Run entire pipeline
multi_model_run(
    train_features=train_features,
    test_features=test_features,
    train_labels=train_labels,
    test_labels=test_labels,
    val_features=val_features,
    val_labels=val_labels,
    reduce_dims="pca",
    model_list=[
        KNeighborsClassifier(n_neighbors=3),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(hidden_layer_sizes=56, max_iter=2000),
    ],
    ####################################
    model_parameters=[
        [[3, 4, 5, 6, 7]],
        [[3, 4, 5], [8, 10, 12], [1, 2, 3]],
        [[28, 42, 56, 70, 84]]
    ],
    ####################################
    metrics=[
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    ],
    res_path=res_path,
    folds=2,
)
# %%
