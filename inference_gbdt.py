import gc
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from cuml.preprocessing.TargetEncoder import TargetEncoder
from pandarallel import pandarallel
from table.src.utils import (
    configure_logger,
    delete_duplicated_columns,
    get_preprocess_parser,
    load_config,
    load_pickle,
    make_submission,
    seed_everything,
    timer,
)
from tqdm import tqdm

if __name__ == "__main__":

    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)

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
        with timer("test"):
            test = pd.read_csv("./dataset/train.csv")

        with timer("Age"):
            age_preds = pd.read_csv(
                f"./preds/{config['features']['age']}/submission.csv"
            )

        with timer("Adoption Speed"):
            adoption_speed_preds = pd.read_csv(
                f"./preds/{config['features']['adoption_speed']}/submission.csv"
            )

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

        with timer("paw_embed"):
            paw_preds = pd.read_csv(
                f"./preds/{config['features']['paw_embed']}/oof_pred.csv"
            )
            paw_preds["Id"] = (
                paw_preds["Id"]
                .str.replace("././dataset//yolox_l_crop/", "")
                .str.replace(".jpg", "")
            )
            paw_preds = paw_preds[
                [col for col in paw_preds.columns if col != "Pawpularity"]
            ]

        with timer("merge"):
            test = pd.merge(test, age_preds, how="left", on="Id")
            test = pd.merge(test, adoption_speed_preds, how="left", on="Id")
            test = pd.merge(test, gender_preds, how="left", on="Id")
            test = pd.merge(test, breed_preds, how="left", on="Id")
            test = pd.merge(test, paw_preds, how="left", on="Id")

            del age_preds, adoption_speed_preds, gender_preds, breed_preds, paw_preds
            gc.collect()

        with timer("oof_train"):
            oof_train = pd.read_csv("output/exp_005/oof_pred.csv")

        categorical_cols = (
            config["categorical_cols"] + test.select_dtypes("category").columns.tolist()
        )
        config["categorical_cols"] = categorical_cols

    with timer("delete duplicated columns"):
        test = delete_duplicated_columns(test)

    with timer("make target and remove cols"):
        y_train = pd.read_csv("./dataset/train.csv")["Pawpularity"].values

        if config["pre_process"]["xentropy"]:
            y_train = y_train / 100
            scaler = None
        elif config["pre_process"]["tweedie"]:
            y_train = 100 - y_train
            scaler = None
        else:
            scaler = None

        cols: List[str] = test.columns.tolist()
        with timer("remove col"):
            remove_cols = ["Id"] + [config["target"]]
            cols = [col for col in cols if col not in remove_cols]
            test = test[cols]

    logging.debug(f"number of test samples: {len(test)}")

    logging.info("Train model")

    with timer("Inference"):
        models = load_pickle("output/exp_005/model.pkl")

        if config["target_encoding"]:
            with timer("target encoding for test"):
                cat_cols = config["categorical_cols"]
                for cat_col in cat_cols:
                    encoder = TargetEncoder(n_folds=4, smooth=0.3)
                    encoder.fit(oof_train[cat_col], y_train)
                    test[cat_col + "_TE"] = encoder.transform(test[cat_col])

        test_preds = np.zeros(len(test))
        for fold in range(10):
            model = models[fold]

            test_preds += (
                model.predict(test.values, ntree_limit=model.best_ntree_limit) / 10
            )

        print("a")
