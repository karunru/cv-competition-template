import atexit
import getpass
import os
import warnings
from pathlib import Path

import mlcrate as mlc
import neptune.new as neptune
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import ttach as tta
from adabelief_pytorch import AdaBelief
from augmentations.augmentation import seti_transform0
from augmentations.strong_aug import *
from cuml.metrics import log_loss  # , roc_auc_score
from dataset import SetiDataset
from fastprogress.fastprogress import force_console_behavior
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from losses import FocalLoss, RocAucLoss, ROCStarLoss
from madgrad import MADGRAD
from models.resnetrs import ResNet_18RS, resnetrs_init_weights
from omegaconf import OmegaConf
from optimizer.sam import SAM
from pandarallel import pandarallel
from sampling import get_sampling
from sklearn.metrics import roc_auc_score
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
        project="karunru/seti",
        api_token=os.environ["NEPTUNE_SETI_API_TOKEN"],
    )

    args = parse_args()
    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(args.options)
    atexit.register(
        remove_abnormal_exp, log_path=config.log_path, config_path=config.config_path
    )
    seed_everything(config.seed)

    exp_num = find_exp_num(log_path=config.log_path)
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

    train_df = pd.read_csv(Path(config.root) / "train_labels.csv")
    train_df["file_name"] = train_df["id"].parallel_apply(
        lambda id: Path(config.root) / f"train/{id[0]}/{id}.npy"
    )
    X = train_df["file_name"].values
    y = train_df["target"].values

    transform = eval(config.transform.name)(config.transform.size)
    logger.info(f"augmentation: {transform}")
    strong_transform = eval(config.strong_transform.name)
    logger.info(f"strong augmentation: {config.strong_transform.name}")

    splits = get_validation(train_df, config)

    scores = np.zeros(len(splits))
    for fold, (train_idx, val_idx) in enumerate(splits):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        X_train, y_train = get_sampling(X_train.reshape(-1, 1), y_train, config)
        X_train = X_train.reshape(-1)
        train_data = SetiDataset("train", X_train, y_train, transform["albu_train"])
        val_data = SetiDataset("val", X_val, y_val, transform["albu_val"])
        train_loader = DataLoader(train_data, **config.train_loader)
        val_loader = DataLoader(val_data, **config.val_loader)

        model = eval(config.model)(pretrained=False)
        if config.model == "ResNet_18RS":
            if config.dino_pretrained_path is not None:
                print(f"load {config.dino_pretrained_path}")
                state_dict = torch.load(
                    config.dino_pretrained_path, map_location="cpu"
                )["teacher"]
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
                state_dict = {
                    k.replace("backbone.", ""): v for k, v in state_dict.items()
                }
                model.load_state_dict(state_dict, strict=False)
            else:
                resnetrs_init_weights(model)
        elif config.model == "resnet18":
            model = torchvision.models.resnet18(pretrained=False)
            state_dict = torch.load(config.dino_pretrained_path, map_location="cpu")[
                "teacher"
            ]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)

        if "fc.weight" in model.state_dict().keys():
            model.fc = nn.Linear(model.fc.in_features, config.train.num_labels)
        elif "classifier.weight" in model.state_dict().keys():
            model.classifier = nn.Linear(
                model.classifier.in_features, config.train.num_labels
            )
        elif "head.fc.weight" in model.state_dict().keys():
            model.head.fc = nn.Linear(
                model.head.fc.in_features, config.train.num_labels
            )
        elif "head.weight" in model.state_dict().keys():
            model.head = nn.Linear(model.head.in_features, config.train.num_labels)

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
            scheduler = eval(config.scheduler.name)(
                optimizer.base_optimizer,
                max(1, config.train.epoch // config.scheduler.cycle),
                eta_min=config.scheduler.eta_min,
            )
        else:
            optimizer = eval(config.optimizer.name)(
                model.parameters(), lr=config.optimizer.lr
            )
            scheduler = eval(config.scheduler.name)(
                optimizer,
                max(1, config.train.epoch // config.scheduler.cycle),
                eta_min=config.scheduler.eta_min,
            )

        criterion = eval(config.loss)()
        scaler = GradScaler()

        best_acc = 0
        best_loss = 1e10
        mb = master_bar(range(config.train.epoch))
        for epoch in mb:
            timer.add("train")
            # if (config.model.simsiam_pretrained_path is not None) and epoch == 5:
            #     model.requires_grad_(True)

            train_loss, train_rmse = train(
                config,
                model,
                transform["torch_train"],
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
            val_loss, val_acc, oof_pred = validate(
                config, model, transform["torch_val"], val_loader, criterion, mb, epoch
            )
            val_time = timer.fsince("val")

            output1 = "epoch: {} train_time: {} validate_time: {}".format(
                epoch, train_time, val_time
            )
            output2 = "train_loss: {:.3f} train_auc: {:.3f} val_loss: {:.3f} val_auc: {:.3f}".format(
                train_loss, train_acc, val_loss, val_acc
            )
            logger.info(output1)
            logger.info(output2)
            mb.write(output1)
            mb.write(output2)
            csv_logger.write([fold, epoch, train_loss, train_acc, val_loss, val_acc])
            run[f"fold_{fold}/train_loss"].log(train_loss)
            run[f"fold_{fold}/train_auc"].log(train_acc)
            run[f"fold_{fold}/val_loss"].log(val_loss)
            run[f"fold_{fold}/val_auc"].log(val_acc)

            scheduler.step()

            if val_loss < best_loss:
                best_loss = val_loss
                save_name = Path(config.weight_path) / f"best_loss_fold{fold}.pth"
                save_model(save_name, epoch, val_loss, val_acc, model, optimizer)
            if val_acc > best_acc:
                best_acc = val_acc
                scores[fold] = best_acc
                train_df.loc[val_idx, "oof_pred"] = oof_pred
                save_name = Path(config.weight_path) / f"best_acc_fold{fold}.pth"
                save_model(save_name, epoch, val_loss, val_acc, model, optimizer)

            save_name = Path(config.weight_path) / f"last_epoch_fold{fold}.pth"
            save_model(save_name, epoch, val_loss, val_acc, model, optimizer)

        del model
        torch.cuda.empty_cache()

    train_df[["id", "target", "oof_pred"]].to_csv(
        Path(config.pred_path) / "oof_pred.csv", index=False
    )

    run["mean_cv_auc"] = np.mean(scores)
    run["oof_auc"] = roc_auc_score(
        train_df["target"].values, train_df["oof_pred"].values
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
    scores = []

    model.train()
    for it, (images, labels) in enumerate(progress_bar(loader, parent=mb)):
        images = images.cuda()
        labels = labels.cuda()
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
                optimizer.zero_grad()

        logits = logits.sigmoid().detach().cpu().numpy().astype(float)
        labels = labels.detach().cpu().numpy().astype(int)
        score = log_loss(labels.reshape(-1), logits.reshape(-1))
        scores.append(score)
        preds.append(logits)
        gt.append(labels)
        losses.append(loss.item())

        mb.child.comment = "loss: {:.3f} avg_loss: {:.3f} log_loss: {:.3f} avg_log_loss: {:.3f}".format(
            loss.item(),
            np.mean(losses),
            score,
            np.mean(scores),
        )

    if config.loss == "ROCStarLoss":
        criterion.last_whole_y_t = torch.tensor(criterion.whole_y_t).cuda()
        criterion.last_whole_y_pred = torch.tensor(criterion.whole_y_pred).cuda()
        criterion.epoch_update_gamma(epoch)

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    score = roc_auc_score(gt.reshape(-1), preds.reshape(-1))
    return np.mean(losses), score


@torch.no_grad()
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
        labels = labels.cuda()
        images = transform(images)

        logits = model(images)
        loss = criterion(logits, labels) / config.train.accumulate

        logits = logits.sigmoid().cpu().numpy()
        labels = labels.cpu().numpy().astype(int)
        preds.append(logits)
        gt.append(labels)
        losses.append(loss.item())

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    score = roc_auc_score(gt.reshape(-1), preds.reshape(-1))

    return np.mean(losses), score, preds


if __name__ == "__main__":
    main()
