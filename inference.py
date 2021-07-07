import atexit
import os
import warnings
from pathlib import Path

import mlcrate as mlc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from augmentations.augmentation import seti_transform0
from augmentations.strong_aug import *
from cuml.metrics import log_loss, roc_auc_score
from dataset import SetiDataset
from fastprogress import master_bar, progress_bar
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from omegaconf import OmegaConf
from pandarallel import pandarallel
from sampling import get_sampling
from timm.models import *
from timm.models.nfnet import *
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import (
    find_exp_num,
    get_logger,
    parse_args,
    remove_abnormal_exp,
    save_model,
    seed_everything,
)
from validation import get_validation

pandarallel.initialize(progress_bar=True)
warnings.filterwarnings("ignore")
tqdm.pandas()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(args.options)
    seed_everything(config.seed)

    exp_num = (
        str(args.config)
        .replace(config.config_path, "")
        .replace("exp_", "")
        .replace(".yaml", "")
    )
    print(exp_num)

    os.makedirs(config.pred_path, exist_ok=True)

    test_df = pd.read_csv(Path(config.root) / "sample_submission.csv")
    test_df["file_name"] = test_df["id"].parallel_apply(
        lambda id: Path(config.root) / f"test/{id[0]}/{id}.npy"
    )
    X = test_df["file_name"].values

    transform = eval(config.transform.name)(config.transform.size)

    test_data = SetiDataset("test", X, y=None, transform=transform["albu_val"])
    test_loader = DataLoader(test_data, **config.val_loader)

    model = eval(config.model)(pretrained=False, in_chans=1)
    if "fc.weight" in model.state_dict().keys():
        model.fc = nn.Linear(model.fc.in_features, config.train.num_labels)
    elif "classifier.weight" in model.state_dict().keys():
        model.classifier = nn.Linear(
            model.classifier.in_features, config.train.num_labels
        )
    elif "head.fc.weight" in model.state_dict().keys():
        model.head.fc = nn.Linear(model.head.fc.in_features, config.train.num_labels)
    model = model.cuda()

    states = [
        (torch.load(Path(config.weight_path) / f"best_acc_fold{fold}.pth")["weight"])
        for fold in range(config.train.n_splits)
    ]

    preds = inference(
        model,
        transform["torch_val"],
        test_loader,
        states,
    )

    test_df["target"] = preds
    test_df[["id", "target"]].to_csv(
        Path(config.pred_path) / f"exp_{exp_num}_2.csv", index=False
    )


@torch.no_grad()
def inference(model, transform, loader, states):
    avg_preds = []

    mb = master_bar(range(len(states)))
    for fold in mb:
        state = states[fold]
        model.load_state_dict(state)
        model.eval()
        preds = []

        for it, images in enumerate(progress_bar(loader, parent=mb)):
            images = images.cuda()
            images = transform(images)

            y_preds = model(images)
            preds.append(y_preds.sigmoid().cpu().numpy())

            mb.child.comment = f"fold: {fold}, iter: {it}"

        preds = pd.Series(np.concatenate(preds).reshape(-1)).rank(pct=True).values
        avg_preds.append(preds)
        mb.main_bar.comment = f"fold: {fold}"
        mb.write(f"finish fold {fold}")

    avg_preds = np.mean(avg_preds, axis=0)

    return avg_preds


if __name__ == "__main__":
    main()
