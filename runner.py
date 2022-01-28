import sklearn.neural_network

from src.backbone import *

# %%
main_path = Path(os.getcwd()) / "data"
time_now = datetime.now().strftime("%d_%m_%Hh%M")
res_path = str(os.getcwd()) + "/outputs/" + time_now
# %%
# READ DATA : Dont forget to remove subset (set to None for full data)
(
    train_features,
    val_features,
    train_labels,
    val_labels,
    test_features,
    test_labels,
) = load_data(main_path=main_path, subset=100)
print_shapes(
    [train_features, val_features, train_labels, val_labels, test_features, test_labels]
)

# %%
# SEE IF READING WORKED + make folder for results
if not Path.is_dir(Path(res_path)):
    os.mkdir(res_path)

start_time = time.time()
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
        KNeighborsClassifier,
        RandomForestClassifier,
        MLPClassifier,
        DecisionTreeClassifier,
    ],
    model_parameters=[
        {"n_neighbors": [3, 4, 5, 6, 7]},
        {
            "max_depth": [3, 4, 5],
            "n_estimators": [8, 10, 12],
            "max_features": [1, 2, 3],
        },
        {"hidden_layer_sizes": [28, 42, 56, 70, 84], "max_iter": [1000, 1500, 2000]},
        {
            "random_state": [0, 1, 2, 3, 4],
            "max_depth": [10, 15, 20, 25],
            "max_features": [10, 15, 20, 25],
        },
    ],
    metrics=[
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    ],
    res_path=res_path,
    folds=5,
)
# %%
# Save run_time to the outputs
end_time = time.time() - start_time
with open(res_path + "/outputs.csv", "a+") as f:
    f.write(f"\nTime taken : {end_time}")
