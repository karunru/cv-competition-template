import atexit
import getpass
import os
import warnings
from pathlib import Path

import mlcrate as mlc
import neptune.new as neptune
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch_optimizer as optim
import torchvision
import ttach as tta
from augmentations.augmentation import pet_transform
from augmentations.strong_aug import *
from cuml.metrics import log_loss
from dataset import PetDataset
from fastprogress.fastprogress import force_console_behavior
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from losses import FocalLoss, RocAucLoss, ROCStarLoss
from models.resnetrs import ResNet_18RS, resnetrs_init_weights
from omegaconf import OmegaConf
from optimizer.sam import SAM
from pandarallel import pandarallel
from sampling import get_sampling
from sklearn.metrics import mean_squared_error, roc_auc_score
from timm.models import *
from timm.models.nfnet import *
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import (
    Mish,
    TanhExp,
    find_exp_num,
    get_logger,
    parse_args,
    remove_abnormal_exp,
    replace_activations,
    save_model,
    seed_everything,
)
from validation import get_validation

pandarallel.initialize(progress_bar=True)
warnings.filterwarnings("ignore")
tqdm.pandas()
master_bar, progress_bar = force_console_behavior()


def main():
    run = neptune.init(
        project="karunru/petfinder-pawpularity-score",
        api_token=os.environ["NEPTUNE_API_TOKEN"],
    )

    args = parse_args()
    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(args.options)
    atexit.register(
        remove_abnormal_exp, log_path=config.log_path, config_path=config.config_path
    )
    seed_everything(config.seed)

    exp_num = find_exp_num(config_path=config.config_path)
    exp_num = str(exp_num).zfill(3)
    run["exp_num"] = exp_num

    config.weight_path = str(Path(config.weight_path) / f"exp_{exp_num}")
    os.makedirs(config.weight_path, exist_ok=True)
    config.pred_path = str(Path(config.pred_path) / f"exp_{exp_num}")
    os.makedirs(config.pred_path, exist_ok=True)
    OmegaConf.save(config, Path(config.config_path) / f"exp_{exp_num}.yaml")
    run["params"] = config
    logger, csv_logger = get_logger(config, exp_num)
    timer = mlc.time.Timer()
    logger.info(mlc.time.now())
    logger.info(f"config: {config}")

    train_df = pd.read_csv(Path(config.root) / "train.csv")
    train_df["filepath"] = "./dataset/crop/" + train_df["Id"] + ".jpg"
    # Sturges' rule
    num_bins = int(np.floor(1 + np.log2(len(train_df))))
    train_df.loc[:, "bins"] = pd.cut(
        train_df["Pawpularity"], bins=num_bins, labels=False
    )
    train_df["oof_pred"] = -1

    X = train_df["filepath"].values
    y = train_df["Pawpularity"].values

    transform = eval(config.transform.name)()
    logger.info(f"augmentation: {transform}")
    strong_transform = eval(config.strong_transform.name)
    logger.info(f"strong augmentation: {config.strong_transform.name}")

    splits = get_validation(train_df, config)

    scores = np.zeros(len(splits))
    for fold, (train_idx, val_idx) in enumerate(splits):
        if not fold in config["train_folds"]:
            continue
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        X_train, y_train = get_sampling(X_train.reshape(-1, 1), y_train, config)
        X_train = X_train.reshape(-1)
        train_data = PetDataset(
            X_train,
            y_train,
            config.transform.size,
        )
        val_data = PetDataset(
            X_val,
            y_val,
            config.transform.size,
        )
        train_loader = DataLoader(train_data, **config.train_loader)
        val_loader = DataLoader(val_data, **config.val_loader)

        model = eval(config.model)(pretrained=False)
        if "fc.weight" in model.state_dict().keys():
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.fc.in_features, config.train.num_labels),
            )
        elif "classifier.weight" in model.state_dict().keys():
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.classifier.in_features, config.train.num_labels),
            )
        elif "head.fc.weight" in model.state_dict().keys():
            model.head.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.head.fc.in_features, config.train.num_labels),
            )
        elif "head.weight" in model.state_dict().keys():
            model.head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.head.in_features, config.train.num_labels),
            )

        model = model.cuda()

        if config.use_SAM:
            optimizer = eval(config.optimizer.name)
            optimizer = SAM(
                model.parameters(),
                base_optimizer=optimizer,
                rho=0.15,
                adaptive=True,
                lr=config.optimizer.lr,
            )
        else:
            optimizer = eval(config.optimizer.name)(
                model.parameters(), lr=config.optimizer.lr
            )

        scheduler = eval(config.scheduler.name)(
            optimizer.base_optimizer if config.use_SAM else optimizer,
            T_0=config.scheduler.T_0,
            eta_min=config.scheduler.eta_min,
        )

        criterion = eval(config.loss.name)()

        scaler = GradScaler()

        best_rmse = 1e10
        best_loss = 1e10
        mb = master_bar(range(config.train.epoch))
        for epoch in mb:
            timer.add("train")

            train_loss, train_rmse = train(
                config,
                model,
                transform["train"],
                strong_transform,
                train_loader,
                optimizer,
                criterion,
                mb,
                epoch,
                scaler,
            )
            train_time = timer.fsince("train")

            timer.add("val")
            val_loss, val_rmse, oof_pred = validate(
                config, model, transform["val"], val_loader, criterion, mb, epoch
            )
            val_time = timer.fsince("val")

            output1 = f"fold: {fold} epoch: {epoch} train_time: {train_time} validate_time: {val_time}"
            output2 = f"train_loss: {train_loss:.3f} train_rmse: {train_rmse:.3f} val_loss: {val_loss:.3f} val_rmse: {val_rmse:.3f}"

            logger.info(output1)
            logger.info(output2)
            mb.write(output1)
            mb.write(output2)
            csv_logger.write([fold, epoch, train_loss, train_rmse, val_loss, val_rmse])
            run[f"fold_{fold}/train_loss"].log(train_loss)
            run[f"fold_{fold}/train_rmse"].log(train_rmse)
            run[f"fold_{fold}/val_loss"].log(val_loss)
            run[f"fold_{fold}/val_rmse"].log(val_rmse)
            run[f"fold_{fold}/learning_rate"].log(scheduler.get_last_lr())

            scheduler.step()

            if val_loss <= best_loss:
                best_loss = val_loss
                save_name = Path(config.weight_path) / f"best_loss_fold{fold}.pth"
                save_model(save_name, epoch, val_loss, val_rmse, model, optimizer)
            if val_rmse <= best_rmse:
                best_rmse = val_rmse
                scores[fold] = best_rmse
                train_df.loc[val_idx, "oof_pred"] = oof_pred
                save_name = Path(config.weight_path) / f"best_acc_fold{fold}.pth"
                save_model(save_name, epoch, val_loss, val_rmse, model, optimizer)

            save_name = Path(config.weight_path) / f"last_epoch_fold{fold}.pth"
            save_model(save_name, epoch, val_loss, val_rmse, model, optimizer)

        del model
        torch.cuda.empty_cache()

    train_df[["Id", "Pawpularity", "oof_pred"]].to_csv(
        Path(config.pred_path) / "oof_pred.csv", index=False
    )

    run["mean_cv_rmse"] = np.mean(scores)

    run["oof_rmse"] = np.sqrt(
        mean_squared_error(train_df["Pawpularity"].values, train_df["oof_pred"].values)
    )


@torch.enable_grad()
def train(
    config,
    model,
    transform,
    strong_transform,
    loader,
    optimizer,
    criterion,
    mb,
    epoch,
    scaler,
):
    preds = []
    gt = []
    losses = []

    model.train()
    for it, (images, labels) in enumerate(progress_bar(loader, parent=mb)):
        images = images.cuda()
        labels = labels.cuda() / 100
        images = transform(images)

        if config.use_SAM:

            #  first step
            if epoch < config.train.epoch - 5:
                with autocast():
                    images, labels_a, labels_b, lam = strong_transform(
                        images, labels, **config.strong_transform.params
                    )
                    logits = model(images)
                    loss = criterion(logits, labels_a) * lam + criterion(
                        logits, labels_b
                    ) * (1 - lam)
                    loss /= config.train.accumulate

                    loss = (loss - config.flooding.b).abs() + config.flooding.b
            else:
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                    loss /= config.train.accumulate

            loss.backward()
            if (it + 1) % config.train.accumulate == 0:
                optimizer.first_step(zero_grad=True)

            # second step
            if epoch < config.train.epoch - 5:
                with autocast():
                    images, labels_a, labels_b, lam = strong_transform(
                        images, labels, **config.strong_transform.params
                    )
                    logits = model(images)
                    loss = criterion(logits, labels_a) * lam + criterion(
                        logits, labels_b
                    ) * (1 - lam)
                    loss /= config.train.accumulate

                    loss = (loss - config.flooding.b).abs() + config.flooding.b
            else:
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                    loss /= config.train.accumulate

            loss.backward()
            if (it + 1) % config.train.accumulate == 0:
                optimizer.second_step(zero_grad=True)
        else:
            if epoch < config.train.epoch - 5:
                with autocast():
                    images, labels_a, labels_b, lam = strong_transform(
                        images, labels, **config.strong_transform.params
                    )
                    logits = model(images)
                    loss = criterion(logits, labels_a) * lam + criterion(
                        logits, labels_b
                    ) * (1 - lam)
                    loss /= config.train.accumulate

                    loss = (loss - config.flooding.b).abs() + config.flooding.b
            else:
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                    loss /= config.train.accumulate

            scaler.scale(loss).backward()
            if (it + 1) % config.train.accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        logits = logits.sigmoid().detach().cpu()
        labels = labels.detach().cpu()
        preds.append(logits)
        gt.append(labels)
        losses.append(loss.item())

        mb.child.comment = "loss: {:.3f} avg_loss: {:.3f}".format(
            loss.item(),
            np.mean(losses),
        )

    preds = np.concatenate(preds) * 100.0
    gt = np.concatenate(gt) * 100.0
    score = np.sqrt(((gt - preds) ** 2).mean())

    return np.mean(losses), score


@torch.inference_mode()
def validate(config, model, transform, loader, criterion, mb, device):
    preds = []
    gt = []
    losses = []

    if config.TTA:
        model = tta.ClassificationTTAWrapper(
            model=model, transforms=tta.aliases.flip_transform(), merge_mode="mean"
        )
    model.eval()

    for it, (images, labels) in enumerate(progress_bar(loader, parent=mb)):
        images = images.cuda()
        labels = labels.cuda() / 100
        images = transform(images)

        logits = model(images)
        loss = criterion(logits, labels) / config.train.accumulate

        logits = logits.sigmoid().detach().cpu()
        labels = labels.detach().cpu()
        preds.append(logits)
        gt.append(labels)
        losses.append(loss.item())

    preds = np.concatenate(preds) * 100.0
    gt = np.concatenate(gt) * 100.0
    score = np.sqrt(((gt - preds) ** 2).mean())

    return np.mean(losses), score, preds


if __name__ == "__main__":
    main()
