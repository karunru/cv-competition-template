import atexit
import gc
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from dataset import PetfinderAgeDataModule
from models.model import Model
from omegaconf import OmegaConf
from pandarallel import pandarallel
from pytorch_lightning import callbacks, seed_everything
from pytorch_lightning.loggers import NeptuneLogger
from sampling import get_sampling
from sklearn.preprocessing import LabelEncoder
from utils import find_exp_num, parse_args, remove_abnormal_exp
from validation import get_validation

pandarallel.initialize(progress_bar=True)
warnings.filterwarnings("ignore")


def main():
    logger = NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project="karunru/petfinder-pawpularity-score-age",
        log_model_checkpoints=False,
    )

    args = parse_args()
    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(args.options)
    atexit.register(
        remove_abnormal_exp, log_path=config.log_path, config_path=config.config_path
    )
    seed_everything(config.seed)
    torch.autograd.set_detect_anomaly(True)

    exp_num = find_exp_num(config_path=config.config_path)
    exp_num = str(exp_num).zfill(3)
    config.exp_num = exp_num

    config.weight_path = str(Path(config.weight_path) / f"exp_{exp_num}")
    os.makedirs(config.weight_path, exist_ok=True)
    config.pred_path = str(Path(config.pred_path) / f"exp_{exp_num}")
    os.makedirs(config.pred_path, exist_ok=True)
    logger.log_hyperparams(config)

    df = pd.read_csv(Path(config.root) / "whole_data.csv")
    df["filepath"] = (
        f"{config.root}/crop/" + df["is_train"] + "_images/" + df["filename"]
    )
    # Sturges' rule
    num_bins = int(np.floor(1 + np.log2(len(df))))
    df.loc[:, "bins"] = pd.cut(df["Age"], bins=num_bins, labels=False)
    le = LabelEncoder()
    df["bins"] = le.fit_transform(df["bins"])
    df["oof_pred"] = -1

    splits = get_validation(df, config)

    scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        if not fold in config["train_folds"]:
            continue

        train_df = df.loc[train_idx].reset_index(drop=True)
        train_df, _ = get_sampling(train_df, train_df[config.val.params.target], config)
        print(f"sampled:")
        print(train_df.groupby(config.val.params.target)["filepath"].count())

        val_df = df.loc[val_idx].reset_index(drop=True)
        datamodule = PetfinderAgeDataModule(train_df, val_df, config)
        model = Model(config)
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping.patience,
        )
        swa = callbacks.StochasticWeightAveraging(
            swa_epoch_start=config.stochastic_weight_avg.swa_epoch_start,
            annealing_epochs=config.stochastic_weight_avg.annealing_epochs,
        )
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            dirpath=config.weight_path,
            filename=f"fold_{fold}_best_loss",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=False,
            save_weights_only=True,
        )
        logger._prefix = f"fold_{fold}"

        trainer = pl.Trainer(
            logger=logger,
            max_epochs=config.epoch,
            callbacks=[lr_monitor, loss_checkpoint, early_stopping, swa],
            **config.trainer,
        )
        # trainer.tune(model, datamodule=datamodule)
        trainer.fit(model, datamodule=datamodule)
        output = trainer.predict(
            model=model,
            dataloaders=datamodule.val_dataloader(),
            return_predictions=True,
            ckpt_path="best",
        )
        oof_pred = []
        for _output in output:
            oof_pred.append(_output[0])
        oof_pred = torch.cat(oof_pred).numpy()
        df.loc[val_idx, "oof_pred"] = oof_pred
        score = np.sqrt(
            (
                (df.loc[val_idx, "oof_pred"] - df.loc[val_idx, "Age"].clip(0, 100)) ** 2
            ).mean()
        )
        scores.append(score)
        logger.experiment[f"fold_{fold}/best_rmse"] = score

        del trainer, model, oof_pred
        gc.collect()
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()

    logger.experiment["mean_cv"] = np.mean(scores)
    logger.experiment["oof_score"] = np.sqrt(
        ((df["oof_pred"] - df["Age"].clip(0, 100)) ** 2).mean()
    )

    df[["filepath", "Age", "oof_pred"]].to_csv(
        Path(config.pred_path) / "oof_pred.csv", index=False
    )

    OmegaConf.save(config, Path(config.config_path) / f"exp_{exp_num}.yaml")


if __name__ == "__main__":
    main()
