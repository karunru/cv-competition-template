import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks, seed_everything

IN_KAGGLE_NOTEBOOK = "KAGGLE_URL_BASE" in set(os.environ.keys())

if IN_KAGGLE_NOTEBOOK:
    # !pip install ../input/omegaconf/omegaconf-2.0.5-py3-none-any.whl
    print()
else:
    from dataset import PetfinderDataModule, PetfinderInferenceDataModule
    from models.model import Model
    from utils import parse_args

from omegaconf import OmegaConf

warnings.filterwarnings("ignore")


def main():
    if IN_KAGGLE_NOTEBOOK:
        config = OmegaConf.load(
            f"../input/petfinder-pawpularity-score-exp-019/exp_019.yaml"
        )
    else:
        args = parse_args()
        config = OmegaConf.load(args.config)

    seed_everything(config.seed)
    torch.autograd.set_detect_anomaly(True)

    test_df = pd.read_csv(
        Path(
            "../input/petfinder-pawpularity-score"
            if IN_KAGGLE_NOTEBOOK
            else config.root
        )
        / "test.csv"
    )
    test_df["Id"] = (
        (
            "../input/petfinder-pawpularity-score/test/"
            if IN_KAGGLE_NOTEBOOK
            else "./dataset/test/"
        )
        + test_df["Id"]
        + ".jpg"
    )

    preds = []
    for fold in config.train_folds:

        datamodule = PetfinderInferenceDataModule(test_df, config)
        model = Model(config)
        model.load_state_dict(
            torch.load(f"{config.weight_path}/fold_{fold}_best_loss.ckpt")["state_dict"]
        )
        model = model.cuda().eval()

        trainer = pl.Trainer(
            logger=False,
            **config.trainer,
        )

        oof_pred = trainer.predict(
            model=model,
            dataloaders=datamodule.predict_dataloader(),
            return_predictions=True,
        )
        preds.append(torch.cat(oof_pred).numpy())

    test_df["Pawpularity"] = np.mean(preds, axis=0)
    submission_df = test_df[["Id", "Pawpularity"]]
    submission_df.to_csv(f"{config.pred_path}/submission.csv", index=False)


if __name__ == "__main__":
    main()
