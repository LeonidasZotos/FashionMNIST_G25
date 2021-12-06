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
) = load_data(main_path=main_path, subset=None)
print_shapes(
    [train_features, val_features, train_labels, val_labels, test_features, test_labels]
)

# %%
# SEE IF READING WORKED + make folder for results 
os.mkdir(res_path)
visualize_image(train_features, res_path)

# %% Run entire pipeline
multi_model_run(
    train_features=train_features,
    test_features=test_features,
    train_labels=train_labels,
    test_labels=test_labels,
    reduce_dims="pca",
    model_list=[
        KNeighborsClassifier(n_neighbors=3),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    ],
    metrics=[
        accuracy_score,
        precision_score,
        recall_score,
    ],
    res_path=res_path
)
# %%
