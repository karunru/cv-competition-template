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
import xfeat
from cuml import TruncatedSVD
from pandarallel import pandarallel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from table.src.evaluation import calc_metric, pr_auc
from table.src.features.modules import (
    DiffGroupbyTransformer,
    GroupbyTransformer,
    RatioGroupbyTransformer,
)
from table.src.models import get_model
from table.src.utils import (
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
from table.src.utils.visualization import plot_pred_density
from table.src.validation import (  # get_validation,
    KarunruSpearmanCorrelationEliminator,
    default_feature_selector,
    remove_correlated_features,
    remove_ks_features,
    select_top_k_features,
)
from table.src.validation.feature_selection import KarunruConstantFeatureEliminator
from tqdm import tqdm
from validation import get_validation
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

    sys.path.append("table/")

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

    config_name = args.config.split("/")[-1].replace(".yml", "")
    output_dir = output_root_dir / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"model output dir: {str(output_dir)}")

    config["model_output_dir"] = str(output_dir)

    # ===============================
    # === Data/Feature Loading
    # ===============================
    with timer("feature loading"):
        with timer("train"):
            train = pd.read_csv("./dataset/train.csv")

        with timer("bbox_info"):
            bbox_info = pd.read_csv("./dataset/yolox_x_crop/bbox_info.csv")[
                [
                    "Id",
                    "height",
                    "width",
                    "aspect_ratio",
                    "area",
                    "num_dog",
                    "num_cat",
                    "num_teddy_bear",
                    "num_person",
                    "num_dog_cat",
                    "num_dog_cat_teddy_bear",
                ]
            ]

        with timer("Age"):
            age_preds = pd.read_csv(
                f"./preds/{config['features']['age']}/submission.csv"
            )
            age_preds["Age_bin"] = pd.cut(
                age_preds["Age"], [-np.inf, 7, 13, 48, np.inf], labels=False
            )
            age_preds["Age_year"] = (age_preds["Age"] // 12).astype(int)
            # age_preds["Age_over_12"] = (age_preds["Age"] > 12).astype(int)

        with timer("Adoption Speed"):
            adoption_speed_preds = pd.read_csv(
                f"./preds/{config['features']['adoption_speed']}/submission.csv"
            )
            # adoption_speed_preds["AdoptionSpeed_bin"] = pd.cut(
            #     adoption_speed_preds["AdoptionSpeed"], [1, 2, 3, 4], labels=False
            # )
            # adoption_speed_preds["AdoptionSpeed_bin_cut"] = pd.cut(
            #     adoption_speed_preds["AdoptionSpeed"], 4, labels=False
            # )

        with timer("Gender"):
            gender_preds = pd.read_csv(
                f"./preds/{config['features']['gender']}/submission.csv"
            )
            gender_preds["gender"] = np.argmax(
                gender_preds[
                    [col for col in gender_preds.keys() if "gender_pred_" in col]
                ].values,
                axis=1,
            )

        with timer("Breed"):
            breed_preds = pd.read_csv(
                f"./preds/{config['features']['breed']}/submission.csv"
            )
            breed_preds["breed"] = np.argmax(
                breed_preds[
                    [col for col in breed_preds.keys() if "breed_pred_" in col]
                ].values,
                axis=1,
            )

        with timer("Maturity Size"):
            maturity_size_preds = pd.read_csv(
                f"./preds/{config['features']['maturity_size']}/submission.csv"
            )
            maturity_size_preds["MaturitySize_bin"] = pd.cut(
                maturity_size_preds["MaturitySize"], 4, labels=False
            )

        if config["features"]["bin_paw"] is not None:
            with timer("bin paw"):
                bin_paw_preds = pd.read_csv(
                    f"./preds/{config['features']['bin_paw']}/oof_pred.csv"
                ).rename(columns={"oof_pred": "bin_paw_oof_pred"})
                bin_paw_preds["bin_paw_oof_pred_bin"] = pd.cut(
                    bin_paw_preds["bin_paw_oof_pred"],
                    3,
                    labels=False,
                )
                bin_paw_preds = bin_paw_preds[
                    [
                        col
                        for col in bin_paw_preds.columns
                        if col not in ["Pawpularity", "bins_paw"]
                    ]
                ]

        with timer("paw_embed"):
            paw_preds = pd.read_csv(
                f"./preds/{config['features']['paw_embed']}/oof_pred.csv"
            )
            paw_preds["Id"] = (
                paw_preds["Id"]
                .str.replace(f"././dataset//{config['dataset']}/", "")
                .str.replace(".jpg", "")
            )
            paw_preds = paw_preds[
                [col for col in paw_preds.columns if col != "Pawpularity"]
            ]
            ordinal_paw_pred_vec_cols = [
                col for col in paw_preds.columns if "ordinal_paw_pred_vec_" in col
            ]
            # tsvd = TruncatedSVD(n_components=8, random_state=1031)
            # paw_preds[
            #     [f"ordinal_paw_pred_vec_svd_{i}" for i in range(8)]
            # ] = tsvd.fit_transform(paw_preds[ordinal_paw_pred_vec_cols].values)
            paw_preds = paw_preds[
                [
                    col
                    for col in paw_preds.columns
                    if col not in ordinal_paw_pred_vec_cols
                ]
            ]

        with timer("merge"):
            train = pd.merge(train, bbox_info, how="left", on="Id")
            train = pd.merge(train, age_preds, how="left", on="Id")
            train = pd.merge(train, adoption_speed_preds, how="left", on="Id")
            train = pd.merge(train, gender_preds, how="left", on="Id")
            train = pd.merge(train, breed_preds, how="left", on="Id")
            train = pd.merge(train, maturity_size_preds, how="left", on="Id")
            train = pd.merge(train, paw_preds, how="left", on="Id")
            if config["features"]["bin_paw"] is not None:
                train = pd.merge(train, bin_paw_preds, how="left", on="Id")

            del (
                age_preds,
                adoption_speed_preds,
                gender_preds,
                breed_preds,
                maturity_size_preds,
                paw_preds,
            )
            if config["features"]["bin_paw"] is not None:
                del bin_paw_preds
            gc.collect()

        with timer("add feats"):
            #     """
            #     "num_dog",
            #     "num_cat",
            #     "num_teddy_bear",
            #     "num_person",
            #     "num_dog_cat",
            #     "num_dog_cat_teddy_bear",
            #     """
            #     train["Age*num_dog"] = train["Age"] * train["num_dog"]
            #     train["Age*num_cat"] = train["Age"] * train["num_cat"]
            #     train["Age*num_dog_cat"] = train["Age"] * train["num_dog_cat"]
            #     train["Age/num_dog"] = train["Age"] / (train["num_dog"] + 1)
            #     train["Age/num_cat"] = train["Age"] / (train["num_cat"] + 1)
            #     train["Age/num_dog_cat"] = train["Age"] / (train["num_dog_cat"] + 1)
            with timer("combi cats"):
                new_cat_df = pd.concat(
                    [
                        xfeat.ConcatCombination(drop_origin=True, r=r).fit_transform(
                            train[config["groupby_keys"]].astype(str).fillna("none")
                        )
                        for r in [
                            2,
                            3,
                            4,
                        ]
                    ],
                    axis="columns",
                )

                for col in new_cat_df.columns:
                    le = LabelEncoder()
                    new_cat_df[col] = le.fit_transform(new_cat_df[col])

                train = pd.concat(
                    [train, new_cat_df],
                    axis="columns",
                )
                combi_cat_cols = new_cat_df.columns.to_list()
                del new_cat_df
                gc.collect()

            groupby_dict = []
            num_var_list = [
                "Age",
                "MaturitySize",
                "AdoptionSpeed",
                "adoption_speed_pred_vec_0",
                "adoption_speed_pred_vec_1",
                "adoption_speed_pred_vec_2",
                "adoption_speed_pred_vec_3",
                "gender_pred_0",
                "gender_pred_1",
                "gender_pred_2",
            ] + [col for col in train.keys() if "breed_pred_" in col]
            num_stats_list = [
                "mean",
                "var",
                "std",
                "min",
                "max",
                "sum",
            ]
            for key in (
                config["groupby_keys"] + combi_cat_cols + ["bin_paw_oof_pred_bin"]
                if config["features"]["bin_paw"] is not None
                else config["groupby_keys"] + combi_cat_cols
            ):
                groupby_dict.append(
                    {
                        "key": [key],
                        "var": num_var_list,
                        "agg": num_stats_list,
                    }
                )
            groupby = GroupbyTransformer(groupby_dict)
            train = groupby.transform(train)

            groupby = DiffGroupbyTransformer(groupby_dict)
            train = groupby.transform(train)
            train = reduce_mem_usage(train)

            groupby = RatioGroupbyTransformer(groupby_dict)
            train = groupby.transform(train)
            train = reduce_mem_usage(train)

        categorical_cols = (
            config["categorical_cols"]
            + train.select_dtypes("category").columns.tolist()
            + combi_cat_cols
            + ["bin_paw_oof_pred_bin"]
        )
        config["categorical_cols"] = categorical_cols

    with timer("delete duplicated columns"):
        train = delete_duplicated_columns(train)

    with timer("make target and remove cols"):
        if config["target"] == "residual":
            y_paw = train["Pawpularity"].values.reshape(-1)
            oof_paw = train["oof_pred"].values.reshape(-1)
            y_train = y_paw - oof_paw
        else:
            y_train = train[config["target"]].values.reshape(-1)

        if config["pre_process"]["xentropy"]:
            y_train = y_train / 100
            scaler = None
        elif config["pre_process"]["tweedie"]:
            y_train = 100 - y_train
            scaler = None
        else:
            scaler = None

        cols: List[str] = train.columns.tolist()
        with timer("remove col"):
            remove_cols = [] + [config["target"]]
            cols = [col for col in cols if col not in remove_cols]
            train = train[cols]

    assert len(train) == len(y_train)
    logging.debug(f"number of features: {len(cols)}")
    logging.debug(f"number of train samples: {len(train)}")

    # ===============================
    # === Feature Selection
    # ===============================
    with timer("Feature Selection"):

        with timer("Feature Selection by ConstantFeatureEliminator"):
            selector = KarunruConstantFeatureEliminator()
            train = selector.fit_transform(train)
            logging.info(f"Removed features : {set(cols) - set(train.columns)}")
            print(f"Removed features : {set(cols) - set(train.columns)}")
            removed_cols = list(set(cols) - set(train.columns))
            cols = train.columns.tolist()

        with timer("Feature Selection by SpearmanCorrelationEliminator"):
            selector = KarunruSpearmanCorrelationEliminator(
                threshold=config["feature_selection"]["SpearmanCorrelation"][
                    "threshold"
                ],
                dry_run=config["feature_selection"]["SpearmanCorrelation"]["dryrun"],
                not_remove_cols=config["feature_selection"]["SpearmanCorrelation"][
                    "not_remove_cols"
                ]
                if config["feature_selection"]["SpearmanCorrelation"][
                    "not_remove_cols"
                ][0]
                != ""
                else [],
                save_path=Path("./table/features"),
            )
            train = selector.fit_transform(train)
            logging.info(f"Removed features : {set(cols) - set(train.columns)}")
            print(f"Removed features : {set(cols) - set(train.columns)}")
            removed_cols += list(set(cols) - set(train.columns))
            cols = train.columns.tolist()

        # with timer("Feature Selection with Kolmogorov-Smirnov statistic"):
        #     if config["feature_selection"]["Kolmogorov-Smirnov"]["do"]:
        #         number_cols = x_train[cols].select_dtypes(include="number").columns
        #         to_remove = remove_ks_features(
        #             x_train[number_cols], x_test[number_cols], number_cols
        #         )
        #         logging.info(f"Removed features : {to_remove}")
        #         print(f"Removed features : {to_remove}")
        #         cols = [col for col in cols if col not in to_remove]

        categorical_cols = [col for col in categorical_cols if col in cols]
        config["categorical_cols"] = categorical_cols
        logging.info(f"categorical_cols : {config['categorical_cols']}")
        print(f"categorical_cols : {config['categorical_cols']}")
        config["removed_cols"] = removed_cols

    logging.info("Train model")

    # get folds
    with timer("Train model"):
        with timer("get validation"):
            # train["Id"] = (
            #     f"./{config['root']}/{config['dataset']}/" + train["Id"] + ".jpg"
            # )
            if config["val"]["params"]["force_recreate"]:
                train[config["val"]["params"]["target"]] = (y_train * 100).astype(int)

            splits = get_validation(train, config)
            cols = [
                col
                for col in cols
                if col not in ["Id", config["val"]["params"]["target"]]
            ]

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
            train_features=train[cols],
            test_features=train[cols].head(),
            y_valid=None,
            valid_features=None,
            feature_name=cols,
            folds_ids=splits,
            target_scaler=scaler,
            config=config,
        )

    # ===============================
    # === Save
    # ===============================
    if config["pre_process"]["xentropy"]:
        y_train = y_train * 100
    elif config["pre_process"]["tweedie"]:
        y_train = 100 - y_train
    plot_pred_density(
        y_train,
        oof_preds,
        save_path=output_dir / "pred_density.png",
    )

    if config["target"] == "residual":
        pred_paw = oof_paw + oof_preds
        plot_pred_density(
            y_paw,
            pred_paw,
            save_path=output_dir / "pred_density_redidual.png",
        )
        oof_score = np.sqrt(np.mean((pred_paw - y_paw) ** 2))
        config["ensemble_oof_score"] = oof_score
        train["xgb_oof_pred"] = oof_preds
        train["ensemble_pred"] = pred_paw
        train["Pawpularity"] = y_paw

        train[
            ["Id", "Pawpularity", "xgb_oof_pred", "ensemble_pred"] + categorical_cols
        ].to_csv(output_dir / "oof_pred.csv", index=False)

    else:
        config["ensemble_oof_score"] = np.sqrt(
            np.mean(((oof_preds * 0.5 + train["oof_pred"] * 0.5) - y_train) ** 2)
        )
        train["xgb_oof_pred"] = oof_preds
        train["ensemble_pred"] = oof_preds * 0.5 + train["xgb_oof_pred"] * 0.5
        train["Pawpularity"] = y_train

        train[
            ["Id", "Pawpularity", "xgb_oof_pred", "ensemble_pred"] + categorical_cols
        ].to_csv(output_dir / "oof_pred.csv", index=False)

    config["eval_results"] = dict()
    for k, v in evals_results.items():
        config["eval_results"][k] = v
    save_path = output_dir / "output.json"
    save_json(config, save_path)

    plot_feature_importance(feature_importance, output_dir / "feature_importance.png")

    save_pickle(models, output_dir / "model.pkl")
    save_pickle(cols, output_dir / "cols.pkl")

    # slack_notify(config_name + "終わったぞ\n" + str(config))
