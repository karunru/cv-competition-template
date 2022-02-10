import numpy as np
import torch
import torch.optim as optim
import torch_optimizer as t_optim
from augmentations.augmentation import get_default_transforms
from augmentations.strong_aug import get_strong_transforms
from losses import FocalLoss
from optimizer.sam import SAM
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_lightning import LightningModule
from timm import create_model
from torch import nn

from .cswin import *
from .ibot_vit import vit_large
from .mae_vit import vit_large_patch16
from .volo import *


class Model(LightningModule):
    def __init__(self, cfg, batch_size=32):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.__build_model()
        self._criterion = eval(self.cfg.loss)()
        self.transform = get_default_transforms()
        self.strong_transform = get_strong_transforms(self.cfg)
        self.save_hyperparameters(cfg)
        if self.cfg.optimizer.use_SAM:
            self.automatic_optimization = False

    def __build_model(self):
        if self.cfg.model.name == "CSWin_64_12211_tiny_224":
            self.backbone = create_model(
                self.cfg.model.name, pretrained=True, num_classes=0, in_chans=3
            )
            num_features = self.backbone.num_features
            self.backbone.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_tiny_224.pth"
                ),
                strict=False,
            )
        elif self.cfg.model.name == "volo_d1":
            self.backbone = create_model(
                self.cfg.model.name, pretrained=True, num_classes=0, in_chans=3
            )
            num_features = self.backbone.num_features
            self.backbone.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://github.com/sail-sg/volo/releases/download/volo_1/d1_224_84.2.pth.tar"
                ),
                strict=False,
            )
        elif self.cfg.model.name == "volo_d5":
            self.backbone = create_model(
                self.cfg.model.name, pretrained=True, num_classes=0, in_chans=3
            )
            num_features = self.backbone.num_features
            self.backbone.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://github.com/sail-sg/volo/releases/download/volo_1/d5_224_86.10.pth.tar"
                ),
                strict=False,
            )
        elif self.cfg.model.name == "ibot_vit_large":
            self.backbone = vit_large()
            self.backbone.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_pt22k/checkpoint_teacher.pth"
                )["state_dict"]
            )
            num_features = self.backbone.num_features
        elif self.cfg.model.name == "mae_vit_large":
            self.backbone = vit_large_patch16(num_classes=0)
            self.backbone.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth"
                )["model"]
            )
            num_features = self.backbone.num_features
        elif "convnext" in self.cfg.model.name:
            self.backbone = create_model(
                self.cfg.model.name,
                pretrained=True,
                in_22k=True,
                num_classes=21841,
                in_chans=3,
            )
            num_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        else:
            self.backbone = create_model(
                self.cfg.model.name, pretrained=True, num_classes=0, in_chans=3
            )
            num_features = self.backbone.num_features

        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, self.cfg.model.output_dim)
        )

    def forward(self, x):
        f = self.backbone(x)
        out = self.fc(f)
        return out

    def training_step(self, batch, batch_idx):
        if self.cfg.optimizer.use_SAM:
            optimizer = self.optimizers()

            # first forward-backward pass
            loss_1, pred_1, labels_1 = self.__share_step(batch, "train")
            self.manual_backward(loss_1)
            if (batch_idx + 1) % self.cfg.optimizer.params.accumulate_grad_batches == 0:
                optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            loss_2, pred_2, labels_2 = self.__share_step(batch, "train")
            self.manual_backward(loss_2)
            if (batch_idx + 1) % self.cfg.optimizer.params.accumulate_grad_batches == 0:
                optimizer.second_step(zero_grad=True)

            return {"loss": loss_1, "pred": pred_1, "labels": labels_1}
        else:
            loss, pred, labels = self.__share_step(batch, "train")
            return {"loss": loss, "pred": pred, "labels": labels}

    def validation_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "val")
        return {"pred": pred, "labels": labels}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        images, labels = batch
        images = self.transform["val"](images)
        embed = self.backbone(images).detach().cpu()
        out = self.forward(images).squeeze(1)
        if self.cfg.model.output_dim == 1:
            out = out.sigmoid().detach().cpu() * 100.0
        else:
            pred = out.sigmoid().detach().cpu().sum(axis=1)
            out = (pred, out.sigmoid().detach().cpu())
        return out, embed

    def __share_step(self, batch, mode):
        images, labels = batch
        labels = labels.float()
        if self.cfg.model.output_dim == 1:
            labels /= 100.0

        images = self.transform[mode](images)

        if torch.rand(1)[0] < 0.5 and mode == "train":
            mix_images, target_a, target_b, lam = self.strong_transform(
                images, labels, **self.cfg.strong_transform.params
            )
            logits = self.forward(mix_images).squeeze(1)
            loss = self._criterion(logits, target_a) * lam + (
                1 - lam
            ) * self._criterion(logits, target_b)
        else:
            logits = self.forward(images).squeeze(1)
            loss = self._criterion(logits, labels)

        if self.cfg.model.output_dim == 1:
            pred = logits.sigmoid().detach().cpu() * 100.0
            labels = labels.detach().cpu() * 100.0
        else:
            pred = logits.sigmoid().detach().cpu().sum(axis=1)
            labels = labels.detach().cpu().sum(axis=1)
        return loss, pred, labels

    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "val")

    def __share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        for out in outputs:
            pred, label = out["pred"], out["labels"]
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        metrics = torch.sqrt(((labels - preds) ** 2).mean())
        self.log(f"{mode}_loss", metrics)

    def check_gradcam(
        self, dataloader, target_layer, target_category, reshape_transform=None
    ):
        cam = GradCAMPlusPlus(
            model=self,
            target_layer=target_layer,
            use_cuda=self.cfg.trainer.gpus,
            reshape_transform=reshape_transform,
        )

        org_images, labels = iter(dataloader).next()
        cam.batch_size = len(org_images)
        images = self.transform["val"](org_images)
        images = images.to(self.device)
        logits = self.forward(images).squeeze(1)
        pred = logits.sigmoid().detach().cpu().numpy() * 100
        labels = labels.cpu().numpy()

        grayscale_cam = cam(
            input_tensor=images, target_category=target_category, eigen_smooth=True
        )
        org_images = org_images.detach().cpu().numpy().transpose(0, 2, 3, 1) / 255.0
        return org_images, grayscale_cam, pred, labels

    def configure_optimizers(self):
        if self.cfg.optimizer.use_SAM:
            optimizer = eval(self.cfg.optimizer.name)
            optimizer = SAM(
                self.parameters(),
                base_optimizer=optimizer,
                rho=0.15,
                adaptive=True,
                lr=self.cfg.optimizer.params.lr,
            )
            scheduler = eval(self.cfg.scheduler.name)(
                optimizer, **self.cfg.scheduler.params
            )
            return [optimizer], [scheduler]

        else:
            optimizer = eval(self.cfg.optimizer.name)(
                self.parameters(), **self.cfg.optimizer.params
            )
            scheduler = eval(self.cfg.scheduler.name)(
                optimizer, **self.cfg.scheduler.params
            )
            return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html#set-grads-to-none
        optimizer.zero_grad(set_to_none=True)


class ClassificationModel(LightningModule):
    def __init__(self, cfg, batch_size=32):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.__build_model()
        self._criterion = eval(self.cfg.loss)()
        self.transform = get_default_transforms()
        self.strong_transform = get_strong_transforms(self.cfg)
        self.save_hyperparameters(cfg)

    def __build_model(self):
        if self.cfg.model.name == "CSWin_64_12211_tiny_224":
            self.backbone = create_model(
                self.cfg.model.name, pretrained=True, num_classes=0, in_chans=3
            )
            num_features = self.backbone.num_features
            self.backbone.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_tiny_224.pth"
                ),
                strict=False,
            )
        elif self.cfg.model.name == "volo_d1":
            self.backbone = create_model(
                self.cfg.model.name, pretrained=True, num_classes=0, in_chans=3
            )
            num_features = self.backbone.num_features
            self.backbone.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://github.com/sail-sg/volo/releases/download/volo_1/d1_224_84.2.pth.tar"
                ),
                strict=False,
            )
        elif self.cfg.model.name == "volo_d5":
            self.backbone = create_model(
                self.cfg.model.name, pretrained=True, num_classes=0, in_chans=3
            )
            num_features = self.backbone.num_features
            self.backbone.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://github.com/sail-sg/volo/releases/download/volo_1/d5_224_86.10.pth.tar"
                ),
                strict=False,
            )
        elif self.cfg.model.name == "ibot_vit_large":
            self.backbone = vit_large()
            self.backbone.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_pt22k/checkpoint_teacher.pth"
                )["state_dict"]
            )
            num_features = self.backbone.num_features
        elif self.cfg.model.name == "mae_vit_large":
            self.backbone = vit_large_patch16(num_classes=0)
            self.backbone.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth"
                )["model"]
            )
            num_features = self.backbone.num_features
        elif "convnext" in self.cfg.model.name:
            self.backbone = create_model(
                self.cfg.model.name,
                pretrained=True,
                in_22k=True,
                num_classes=21841,
                in_chans=3,
            )
            num_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        else:
            self.backbone = create_model(
                self.cfg.model.name, pretrained=True, num_classes=0, in_chans=3
            )
            num_features = self.backbone.num_features

        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, self.cfg.model.output_dim)
        )

    def forward(self, x):
        f = self.backbone(x)
        out = self.fc(f)
        return out

    def training_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "train")
        return {"loss": loss, "pred": pred, "labels": labels}

    def validation_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "val")
        return {"pred": pred, "labels": labels}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        images, labels = batch
        images = self.transform["val"](images)
        out = self.forward(images).squeeze(1)
        out = out.sigmoid().detach().cpu()
        return out

    def __share_step(self, batch, mode):
        images, labels = batch
        labels = labels.float()

        images = self.transform[mode](images)

        if torch.rand(1)[0] < 0.5 and mode == "train":
            mix_images, target_a, target_b, lam = self.strong_transform(
                images, labels, **self.cfg.strong_transform.params
            )
            logits = self.forward(mix_images).squeeze(1)
            loss = self._criterion(logits, target_a) * lam + (
                1 - lam
            ) * self._criterion(logits, target_b)
        else:
            logits = self.forward(images).squeeze(1)
            loss = self._criterion(logits, labels)

        pred = logits.sigmoid().detach().cpu()
        labels = labels.detach().cpu()
        return loss, pred, labels

    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "val")

    def __share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        for out in outputs:
            pred, label = out["pred"], out["labels"]
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        metrics = torch.nn.BCEWithLogitsLoss()(
            preds.to(torch.float32), labels.to(torch.float32)
        )
        self.log(f"{mode}_loss", metrics)

    def check_gradcam(
        self, dataloader, target_layer, target_category, reshape_transform=None
    ):
        cam = GradCAMPlusPlus(
            model=self,
            target_layer=target_layer,
            use_cuda=self.cfg.trainer.gpus,
            reshape_transform=reshape_transform,
        )

        org_images, labels = iter(dataloader).next()
        cam.batch_size = len(org_images)
        images = self.transform["val"](org_images)
        images = images.to(self.device)
        logits = self.forward(images).squeeze(1)
        pred = logits.sigmoid().detach().cpu().numpy() * 100
        labels = labels.cpu().numpy()

        grayscale_cam = cam(
            input_tensor=images, target_category=target_category, eigen_smooth=True
        )
        org_images = org_images.detach().cpu().numpy().transpose(0, 2, 3, 1) / 255.0
        return org_images, grayscale_cam, pred, labels

    def configure_optimizers(self):
        optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        scheduler = eval(self.cfg.scheduler.name)(
            optimizer, **self.cfg.scheduler.params
        )
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html#set-grads-to-none
        optimizer.zero_grad(set_to_none=False)


class WithTableModel(LightningModule):
    def __init__(self, cfg, batch_size=32):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.__build_model()
        self._criterion = eval(self.cfg.loss)()
        self.transform = get_default_transforms()
        self.strong_transform = get_strong_transforms(self.cfg)
        self.save_hyperparameters(cfg)

    def __build_model(self):
        self.backbone = create_model(
            self.cfg.model.name, pretrained=True, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        num_table_features = self.cfg.model.num_table_features
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features + num_table_features, self.cfg.model.output_dim),
        )

    def forward(self, image, features):
        f = self.backbone(image)
        f = torch.cat([f, features], dim=1)
        out = self.fc(f)
        return out

    def training_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "train")
        return {"loss": loss, "pred": pred, "labels": labels}

    def validation_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "val")
        return {"pred": pred, "labels": labels}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        images, table_features, labels = batch
        images = self.transform["val"](images)
        out = self.forward(images, table_features).squeeze(1)
        if self.cfg.model.output_dim == 1:
            out = out.sigmoid().detach().cpu() * 100.0
        else:
            out = out.sigmoid().detach().cpu().sum(axis=1)
        return out

    def __share_step(self, batch, mode):
        images, table_features, labels = batch
        labels = labels.float()
        if self.cfg.model.output_dim == 1:
            labels /= 100.0

        images = self.transform[mode](images)

        if torch.rand(1)[0] < 0.5 and mode == "train":
            mix_images, target_a, target_b, lam = self.strong_transform(
                images, labels, **self.cfg.strong_transform.params
            )
            logits = self.forward(mix_images, table_features).squeeze(1)
            loss = self._criterion(logits, target_a) * lam + (
                1 - lam
            ) * self._criterion(logits, target_b)
        else:
            logits = self.forward(images, table_features).squeeze(1)
            loss = self._criterion(logits, labels)

        if self.cfg.model.output_dim == 1:
            pred = logits.sigmoid().detach().cpu() * 100.0
            labels = labels.detach().cpu() * 100.0
        else:
            pred = logits.sigmoid().detach().cpu().sum(axis=1)
            labels = labels.detach().cpu().sum(axis=1)
        return loss, pred, labels

    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "val")

    def __share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        for out in outputs:
            pred, label = out["pred"], out["labels"]
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        metrics = torch.sqrt(((labels - preds) ** 2).mean())
        self.log(f"{mode}_loss", metrics)

    def check_gradcam(
        self, dataloader, target_layer, target_category, reshape_transform=None
    ):
        cam = GradCAMPlusPlus(
            model=self,
            target_layer=target_layer,
            use_cuda=self.cfg.trainer.gpus,
            reshape_transform=reshape_transform,
        )

        org_images, labels = iter(dataloader).next()
        cam.batch_size = len(org_images)
        images = self.transform["val"](org_images)
        images = images.to(self.device)
        logits = self.forward(images).squeeze(1)
        pred = logits.sigmoid().detach().cpu().numpy() * 100
        labels = labels.cpu().numpy()

        grayscale_cam = cam(
            input_tensor=images, target_category=target_category, eigen_smooth=True
        )
        org_images = org_images.detach().cpu().numpy().transpose(0, 2, 3, 1) / 255.0
        return org_images, grayscale_cam, pred, labels

    def configure_optimizers(self):
        optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        scheduler = eval(self.cfg.scheduler.name)(
            optimizer, **self.cfg.scheduler.params
        )
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html#set-grads-to-none
        optimizer.zero_grad(set_to_none=False)


class Feats1kModel(LightningModule):
    def __init__(self, cfg, batch_size=32):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.__build_model()
        self._criterion = eval(self.cfg.loss)()
        self.transform = get_default_transforms()
        self.strong_transform = get_strong_transforms(self.cfg)
        self.save_hyperparameters(cfg)

    def __build_model(self):
        self.model = create_model(self.cfg.model.name, pretrained=True, in_chans=3)

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        images, labels = batch
        images = self.transform["val"](images)
        out = self.forward(images).cpu().numpy()
        return out

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html#set-grads-to-none
        optimizer.zero_grad(set_to_none=True)
