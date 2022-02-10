import atexit
import gc
import os
import warnings
from collections import OrderedDict
from glob import glob
from pathlib import Path

import models.convnext
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from dataset import PetfinderDataModule
from models.ibot_vit import vit_large
from models.mae_vit import vit_large_patch16
from models.model import Model
from omegaconf import OmegaConf
from pandarallel import pandarallel
from pytorch_lightning import callbacks, seed_everything
from pytorch_lightning.loggers import NeptuneLogger
from sampling import get_sampling
from timm import create_model
from utils import find_exp_num, parse_args, remove_abnormal_exp
from validation import get_validation

pandarallel.initialize(progress_bar=True)
warnings.filterwarnings("ignore")


def main():
    logger = NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project="karunru/petfinder-pawpularity-score",
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

    df = pd.read_csv(Path(config.root) / "train.csv")
    df["Id"] = f"./{config.root}/{config.dataset}/" + df["Id"] + ".jpg"
    # Sturges' rule
    num_bins = int(np.floor(1 + np.log2(len(df))))
    df.loc[:, "bins"] = pd.cut(df["Pawpularity"], bins=num_bins, labels=False)
    df["oof_pred"] = -1
    if config.model.name == "ibot_vit_large":
        num_features = vit_large().num_features
    elif config.model.name == "mae_vit_large":
        num_features = vit_large_patch16(num_classes=0).num_features
    elif config.model.name == "convnext_large":
        num_features = 1536
    elif config.model.name == "convnext_xlarge":
        num_features = 2048
    else:
        num_features = create_model(
            config.model.name, pretrained=False, num_classes=0, in_chans=3
        ).num_features
    emb_cols = [f"emb_{i}" for i in range(num_features)]
    df.loc[:, emb_cols] = -1
    if config.model.output_dim == 100:
        pred_vec_cols = [
            f"ordinal_paw_pred_vec_{i}" for i in range(config.model.output_dim)
        ]
        df.loc[:, pred_vec_cols] = -1

    splits = get_validation(df, config)

    scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        if not fold in config["train_folds"]:
            continue

        train_df = df.loc[train_idx].reset_index(drop=True)
        train_df, _ = get_sampling(train_df, train_df[config.val.params.target], config)
        print(f"sampled:")
        print(train_df.groupby(config.val.params.target)["Id"].count())

        val_df = df.loc[val_idx].reset_index(drop=True)
        datamodule = PetfinderDataModule(train_df, val_df, config)
        model = Model(config)

        if config.model.pretrained_weight is not None:
            if (config.model.pretrained_weight).startswith("exp_"):
                pretrained_weights_pathes = glob(
                    f"./weights/{config.model.pretrained_weight}/*.ckpt"
                )
                pretrained_weights = [
                    torch.load(weight_path, map_location="cpu")["state_dict"]
                    for weight_path in pretrained_weights_pathes
                ]
            else:
                pretrained_weights_pathes = glob(
                    f"./weights/{config.model.pretrained_weight}/*.pth"
                )
                pretrained_weights = [
                    torch.load(weight_path, map_location="cpu")
                    for weight_path in pretrained_weights_pathes
                ]

            pretrained_weight = pretrained_weights[0].copy()
            for key in pretrained_weight.keys():
                if "relative_position_index" in key:
                    pretrained_weight[key] = pretrained_weights[0][key].clone().detach()
                else:
                    pretrained_weight[key] = torch.mean(
                        torch.stack(
                            [
                                pretrained_weights[i][key]
                                for i in range(len(pretrained_weights))
                            ]
                        ),
                        dim=0,
                    )
            del pretrained_weights

            if (config.model.pretrained_weight).startswith("jinkaido"):
                pretrained_weight = {
                    k.replace("model", "backbone"): v
                    for k, v in pretrained_weight.items()
                }
                pretrained_weight = {
                    k.replace("backbone.head", "fc.1"): v
                    for k, v in pretrained_weight.items()
                }

            model.load_state_dict(
                OrderedDict(
                    {k: v for k, v in pretrained_weight.items() if "fc." not in k}
                ),
                strict=False,
            )

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

        if config.optimizer.use_SAM:
            config.trainer.accumulate_grad_batches = None

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
        emb = []
        oof_pred_vec = []
        for _output in output:
            if config.model.output_dim == 100:
                oof_pred.append(_output[0][0])
                oof_pred_vec.append(_output[0][1])
            else:
                oof_pred.append(_output[0])

            emb.append(_output[1])
        oof_pred = torch.cat(oof_pred).numpy()
        emb = torch.cat(emb).numpy()
        df.loc[val_idx, "oof_pred"] = oof_pred
        df.loc[val_idx, emb_cols] = emb
        if config.model.output_dim == 100:
            oof_pred_vec = torch.cat(oof_pred_vec).numpy()
            df.loc[val_idx, pred_vec_cols] = oof_pred_vec
        score = np.sqrt(
            ((df.loc[val_idx, "oof_pred"] - df.loc[val_idx, "Pawpularity"]) ** 2).mean()
        )
        scores.append(score)
        logger.experiment[f"fold_{fold}/best_rmse"] = score

        del trainer, model, oof_pred
        gc.collect()
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()

    logger.experiment["mean_cv"] = np.mean(scores)
    logger.experiment["oof_score"] = np.sqrt(
        ((df["oof_pred"] - df["Pawpularity"]) ** 2).mean()
    )

    save_cols = ["Id", "Pawpularity", "oof_pred"] + emb_cols
    if config.model.output_dim == 100:
        save_cols += pred_vec_cols

    df[save_cols].to_csv(Path(config.pred_path) / "oof_pred.csv", index=False)

    OmegaConf.save(config, Path(config.config_path) / f"exp_{exp_num}.yaml")


if __name__ == "__main__":
    main()
