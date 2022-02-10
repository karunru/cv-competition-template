import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from dataset import PetfinderAgeInferenceDataModule
from models.model import Model
from omegaconf import OmegaConf
from pytorch_lightning import callbacks, seed_everything
from utils import parse_args

warnings.filterwarnings("ignore")


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(args.options)

    seed_everything(config.seed)
    torch.autograd.set_detect_anomaly(True)

    config.root = "./dataset/"
    test_df = pd.read_csv(Path(config.root) / "train.csv")
    test_df["filepath"] = config.root + "yolox_l_crop/" + test_df["Id"] + ".jpg"

    preds = []
    for fold in config.train_folds:

        datamodule = PetfinderAgeInferenceDataModule(test_df, config)
        model = Model(config)
        model.load_state_dict(
            torch.load(f"{config.weight_path}/fold_{fold}_best_loss.pth")
        )
        model = model.cuda().eval()

        trainer = pl.Trainer(
            logger=False,
            **config.trainer,
        )

        output = trainer.predict(
            model=model,
            dataloaders=datamodule.predict_dataloader(),
            return_predictions=True,
        )
        oof_pred = []
        for _output in output:
            oof_pred.append(_output[0])
        oof_pred = torch.cat(oof_pred)
        preds.append(oof_pred.numpy())

    test_df["Age"] = np.mean(preds, axis=0)
    submission_df = test_df[["Id", "Age"]]
    submission_df.to_csv(f"{config.pred_path}/submission.csv", index=False)


if __name__ == "__main__":
    main()
