import datetime
import gc
import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import List

import cudf
import cupy as cp
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rmm
import seaborn as sns
from pandarallel import pandarallel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from src.evaluation import calc_metric, pr_auc
from src.models import get_model
from src.utils import (
    configure_logger,
    delete_duplicated_columns,
    feature_existence_checker,
    get_preprocess_parser,
    load_config,
    load_pickle,
    make_submission,
    merge_by_concat,
    plot_feature_importance,
    reduce_mem_usage,
    save_json,
    save_pickle,
    seed_everything,
    slack_notify,
    timer,
)
from src.utils.visualization import plot_pred_density
from src.validation import (  # get_validation,
    KarunruSpearmanCorrelationEliminator,
    default_feature_selector,
    remove_correlated_features,
    remove_ks_features,
    select_top_k_features,
)
from src.validation.feature_selection import KarunruConstantFeatureEliminator
from tqdm import tqdm
from xfeat import (
    ConstantFeatureEliminator,
    DuplicatedFeatureEliminator,
    SpearmanCorrelationEliminator,
)

if __name__ == "__main__":
    # Set RMM to allocate all memory as managed memory (cudaMallocManaged underlying allocator)
    rmm.reinitialize(managed_memory=True)
    assert rmm.is_initialized()

    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)

    sys.path.append("./")

    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    warnings.filterwarnings("ignore")

    parser = get_preprocess_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logger(args.config, log_dir=args.log_dir, debug=args.debug)

    seed_everything(config["seed_everything"])

    logging.info(f"config: {args.config}")
    logging.info(f"debug: {args.debug}")

    config["args"] = dict()
    config["args"]["config"] = args.config

    # make output dir
    output_root_dir = Path(config["output_dir"])
    feature_dir = Path(config["dataset"]["feature_dir"])

    config_name = args.config.split("/")[-1].replace(".yml", "")
    output_dir = output_root_dir / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"model output dir: {str(output_dir)}")

    config["model_output_dir"] = str(output_dir)

    # ===============================
    # === Data/Feature Loading
    # ===============================
    input_dir = Path(config["dataset"]["dir"])

    with timer("feature loading"):
        with timer("train"):
            train = pd.read_csv("../dataset/train.csv")

        with timer("Age"):
            age_preds = pd.read_csv("../preds/exp_087/submission.csv")

        with timer("Adoption Speed"):
            adoption_speed_preds = pd.read_csv("../preds/exp_087/submission.csv")

        with timer("Gender"):
            gender_preds = pd.read_csv("../preds/exp_087/submission.csv")
            gender_preds["gender"] = np.argmax(
                gender_preds[
                    [col for col in gender_preds.keys() if "gender_pred_" in col]
                ].values,
                axis=1,
            ).astype("category")

        with timer("Breed"):
            breed_preds = pd.read_csv("../preds/exp_087/submission.csv")
            breed_preds["breed"] = np.argmax(
                breed_preds[
                    [col for col in breed_preds.keys() if "breed_pred_" in col]
                ].values,
                axis=1,
            ).astype("category")

        with timer("paw_embed"):
            paw_preds = pd.read_csv("../preds/exp_087/oof_pred.csv")
            paw_preds["Id"] = (
                paw_preds["Id"]
                .str.replace("././dataset//yolox_l_crop/", "")
                .str.replace(".jpg", "")
            )
            paw_preds = paw_preds[
                [col for col in paw_preds.columns if col != "Pawpularity"]
            ]

        with timer("merge"):
            train = pd.merge(train, age_preds, how="left", on="Id")
            train = pd.merge(train, adoption_speed_preds, how="left", on="Id")
            train = pd.merge(train, gender_preds, how="left", on="Id")
            train = pd.merge(train, breed_preds, how="left", on="Id")

        categorical_cols = (
            config["categorical_cols"]
            + train.select_dtypes("category").columns.tolist()
        )

    with timer("delete duplicated columns"):
        train = delete_duplicated_columns(train)

    if config["stacking"]["do"]:
        with timer("load predictions"):
            org_cols = train.columns.to_list()
            preds = config["stacking"]["predictions"]
            for pred in preds:
                train[pred] = np.load("output/" + pred + "/oof_preds.npy")

    with timer("make target and remove cols"):
        y_train = train[config["target"]].values.reshape(-1)

        if config["pre_process"]["xentropy"]:
            y_train /= 100
            scaler = None
        else:
            scaler = None

        cols: List[str] = train.columns.tolist()
        with timer("remove col"):
            remove_cols = [] + [config["target"]]
            if config["stacking"]["do"]:
                if not config["stacking"]["use_org_cols"]:
                    remove_cols += org_cols
            cols = [col for col in cols if col not in remove_cols]
            train = train[cols]

    assert len(train) == len(y_train)
    logging.debug(f"number of features: {len(cols)}")
    logging.debug(f"number of train samples: {len(train)}")

    # ===============================
    # === Feature Selection
    # ===============================
    # with timer("Feature Selection"):
    #     if not config["stacking"]["do"]:
    #         if config["feature_selection"]["top_k"]["do"]:
    #             use_cols = select_top_k_features(config["feature_selection"]["top_k"])
    #             use_cols = [col for col in use_cols if "_TE" not in col]
    #             use_cols.append("PlaceID")
    #             x_train, x_test = x_train[use_cols], x_test[use_cols]
    #         else:
    #             with timer("Feature Selection by ConstantFeatureEliminator"):
    #                 selector = KarunruConstantFeatureEliminator()
    #                 x_train = selector.fit_transform(x_train)
    #                 x_test = selector.transform(x_test)
    #                 assert len(x_train.columns) == len(x_test.columns)
    #                 logging.info(
    #                     f"Removed features : {set(cols) - set(x_train.columns)}"
    #                 )
    #                 print(f"Removed features : {set(cols) - set(x_train.columns)}")
    #                 cols = x_train.columns.tolist()
    #
    #             with timer("Feature Selection by SpearmanCorrelationEliminator"):
    #                 selector = KarunruSpearmanCorrelationEliminator(
    #                     threshold=config["feature_selection"]["SpearmanCorrelation"][
    #                         "threshold"
    #                     ],
    #                     dry_run=config["feature_selection"]["SpearmanCorrelation"][
    #                         "dryrun"
    #                     ],
    #                     not_remove_cols=config["feature_selection"][
    #                         "SpearmanCorrelation"
    #                     ]["not_remove_cols"]
    #                     if config["feature_selection"]["SpearmanCorrelation"][
    #                         "not_remove_cols"
    #                     ][0]
    #                     != ""
    #                     else [],
    #                 )
    #                 x_train = selector.fit_transform(x_train)
    #                 x_test = selector.transform(x_test)
    #                 assert len(x_train.columns) == len(x_test.columns)
    #                 logging.info(
    #                     f"Removed features : {set(cols) - set(x_train.columns)}"
    #                 )
    #                 print(f"Removed features : {set(cols) - set(x_train.columns)}")
    #                 cols = x_train.columns.tolist()
    #
    #             with timer("Feature Selection with Kolmogorov-Smirnov statistic"):
    #                 if config["feature_selection"]["Kolmogorov-Smirnov"]["do"]:
    #                     number_cols = (
    #                         x_train[cols].select_dtypes(include="number").columns
    #                     )
    #                     to_remove = remove_ks_features(
    #                         x_train[number_cols], x_test[number_cols], number_cols
    #                     )
    #                     logging.info(f"Removed features : {to_remove}")
    #                     print(f"Removed features : {to_remove}")
    #                     cols = [col for col in cols if col not in to_remove]
    #
    #     cols = x_train.columns.tolist()
    #     categorical_cols = [col for col in categorical_cols if col in cols]
    #     config["categorical_cols"] = categorical_cols
    #     logging.info(f"categorical_cols : {config['categorical_cols']}")
    #     print(f"categorical_cols : {config['categorical_cols']}")

    logging.info("Train model")

    # get folds
    with timer("Train model"):
        with timer("get validation"):
            x_train["binned_target"] = pd.qcut(
                y_train,
                q=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
                labels=False,
                duplicates="drop",
            )
            splits = get_validation(x_train, config)
            del x_train["binned_target"]
            gc.collect()

        model = get_model(config)
        (
            models,
            oof_preds,
            test_preds,
            valid_preds,
            feature_importance,
            evals_results,
        ) = model.cv(
            y_train=y_train,
            train_features=x_train[cols],
            test_features=x_test[cols],
            y_valid=None,
            valid_features=None,
            feature_name=cols,
            folds_ids=splits,
            target_scaler=scaler,
            config=config,
        )

    # ===============================
    # === Make submission
    # ===============================

    sample_submission = pd.read_csv(input_dir / "sample_submission.csv")
    submission_df = make_submission(test_preds, sample_submission)

    # ===============================
    # === Save
    # ===============================

    config["eval_results"] = dict()
    for k, v in evals_results.items():
        config["eval_results"][k] = v
    save_path = output_dir / "output.json"
    save_json(config, save_path)

    plot_feature_importance(feature_importance, output_dir / "feature_importance.png")

    plot_pred_density(
        np.log1p(np.expm1(y_train) * x_train[config["pre_process"]["col"]])
        if config["pre_process"]["do"]
        else y_train,
        np.log1p(oof_preds),
        np.log1p(test_preds),
        output_dir / "pred_density.png",
    )

    np.save(output_dir / "oof_preds.npy", oof_preds)

    np.save(output_dir / "test_preds.npy", test_preds)

    submission_df.to_csv(output_dir / f"{config_name}_sub.csv", index=False)

    save_pickle(models, output_dir / "model.pkl")

    # slack_notify(config_name + "終わったぞ\n" + str(config))
