import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from timm.data import create_transform
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image


class PetDataset(Dataset):
    def __init__(self, X, y=None, image_size="224,224"):
        self.X = X
        self.y = y
        size = list(map(int, image_size.split(",")))
        self.transform = T.Resize([size[0], size[1]])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image_path = self.X[idx]

        image = read_image(image_path)
        image = self.transform(image)

        label = (
            torch.tensor(self.y[idx]).float().unsqueeze(-1)
            if self.y is not None
            else None
        )
        if label is not None:
            return image, label
        else:
            return image


class PetfinderDataset(Dataset):
    def __init__(
        self,
        df,
        image_size=224,
        output_dim=1,
        over_100_or_not=False,
        over_50_or_not=False,
    ):
        self._X = df["Id"].values
        self._y = None
        if "Pawpularity" in df.keys():
            pawpularity = df["Pawpularity"].values

            if output_dim == 100:
                self._y = np.zeros((len(df), output_dim)).astype(np.float32)
                for i in range(len(df)):
                    self._y[i, : pawpularity[i]] = 1
            else:
                if over_100_or_not:
                    self._y = (pawpularity == 100).astype(np.float32)
                elif over_50_or_not:
                    self._y = (pawpularity >= 50).astype(np.float32)
                else:
                    self._y = pawpularity

        self._transform = T.Resize([image_size, image_size])

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path)
        image = self._transform(image)
        if self._y is not None:
            label = self._y[idx]
            return image, label
        return image, 0


class PetfinderDataModule(LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        cfg,
    ):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._cfg = cfg

    def __create_dataset(self, train=True):
        return (
            PetfinderDataset(
                self._train_df,
                self._cfg.transform.image_size,
                self._cfg.model.output_dim,
                self._cfg.model.over_100_or_not,
                self._cfg.model.over_50_or_not,
            )
            if train
            else PetfinderDataset(
                self._val_df,
                self._cfg.transform.image_size,
                self._cfg.model.output_dim,
                self._cfg.model.over_100_or_not,
                self._cfg.model.over_50_or_not,
            )
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._cfg.val_loader)


class PetfinderInferenceDataModule(LightningDataModule):
    def __init__(
        self,
        test_df,
        cfg,
    ):
        super().__init__()
        self._test_df = test_df
        self._cfg = cfg

    def predict_dataloader(self):
        dataset = PetfinderDataset(
            self._test_df,
            self._cfg.transform.image_size,
            self._cfg.model.output_dim,
            self._cfg.model.over_100_or_not,
            self._cfg.model.over_50_or_not,
        )
        return DataLoader(dataset, **self._cfg.val_loader)


class PetfinderAgeDataset(Dataset):
    def __init__(self, df, image_size=224, output_dim=1):
        self._X = df["filepath"].values
        self._y = None
        if "Age" in df.keys():
            self._y = df["Age"].clip(0, 100).values

        self._transform = T.Resize([image_size, image_size])

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path, mode=ImageReadMode.RGB)
        image = self._transform(image)
        if self._y is not None:
            label = self._y[idx]
            return image, label
        return image, 0


class PetfinderAgeDataModule(LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        cfg,
    ):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._cfg = cfg

    def __create_dataset(self, train=True):
        return (
            PetfinderAgeDataset(
                self._train_df,
                self._cfg.transform.image_size,
                self._cfg.model.output_dim,
            )
            if train
            else PetfinderAgeDataset(
                self._val_df,
                self._cfg.transform.image_size,
                self._cfg.model.output_dim,
            )
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._cfg.val_loader)


class PetfinderAgeInferenceDataModule(LightningDataModule):
    def __init__(
        self,
        test_df,
        cfg,
    ):
        super().__init__()
        self._test_df = test_df
        self._cfg = cfg

    def predict_dataloader(self):
        dataset = PetfinderAgeDataset(
            self._test_df,
            self._cfg.transform.image_size,
            self._cfg.model.output_dim,
        )
        return DataLoader(dataset, **self._cfg.val_loader)


class PetfinderBreedDataset(Dataset):
    def __init__(self, df, image_size=224, output_dim=100):
        self._X = df["filepath"].values
        self._y = None
        if "Breed1" in df.keys():
            self._y = np.identity(output_dim)[df["Breed1"].values]

        self._transform = T.Resize([image_size, image_size])

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path, mode=ImageReadMode.RGB)
        image = self._transform(image)
        if self._y is not None:
            label = self._y[idx]
            return image, label
        return image, 0


class PetfinderBreedDataModule(LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        cfg,
    ):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._cfg = cfg

    def __create_dataset(self, train=True):
        return (
            PetfinderBreedDataset(
                self._train_df,
                self._cfg.transform.image_size,
                self._cfg.model.output_dim,
            )
            if train
            else PetfinderBreedDataset(
                self._val_df,
                self._cfg.transform.image_size,
                self._cfg.model.output_dim,
            )
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._cfg.val_loader)


class PetfinderBreedInferenceDataModule(LightningDataModule):
    def __init__(
        self,
        test_df,
        cfg,
    ):
        super().__init__()
        self._test_df = test_df
        self._cfg = cfg

    def predict_dataloader(self):
        dataset = PetfinderBreedDataset(
            self._test_df,
            self._cfg.transform.image_size,
            self._cfg.model.output_dim,
        )
        return DataLoader(dataset, **self._cfg.val_loader)


class PetfinderGenderDataset(Dataset):
    def __init__(self, df, image_size=224, output_dim=3):
        self._X = df["filepath"].values
        self._y = None
        if "Gender" in df.keys():
            self._y = np.identity(output_dim)[df["Gender"].values]

        self._transform = T.Resize([image_size, image_size])

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path, mode=ImageReadMode.RGB)
        image = self._transform(image)
        if self._y is not None:
            label = self._y[idx]
            return image, label
        return image, 0


class PetfinderGenderDataModule(LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        cfg,
    ):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._cfg = cfg

    def __create_dataset(self, train=True):
        return (
            PetfinderGenderDataset(
                self._train_df,
                self._cfg.transform.image_size,
            )
            if train
            else PetfinderGenderDataset(
                self._val_df,
                self._cfg.transform.image_size,
            )
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._cfg.val_loader)


class PetfinderGenderInferenceDataModule(LightningDataModule):
    def __init__(
        self,
        test_df,
        cfg,
    ):
        super().__init__()
        self._test_df = test_df
        self._cfg = cfg

    def predict_dataloader(self):
        dataset = PetfinderGenderDataset(self._test_df, self._cfg.transform.image_size)
        return DataLoader(dataset, **self._cfg.val_loader)


class PetfinderWithTableDataset(Dataset):
    def __init__(
        self, df, table_cols, image_size=224, output_dim=1, over_100_or_not=False
    ):
        self._X = df["Id"].values
        self._table = df[table_cols].values.astype(np.float32)
        self._y = None
        if "Pawpularity" in df.keys():
            pawpularity = df["Pawpularity"].values

            if output_dim == 100:
                self._y = np.zeros((len(df), output_dim)).astype(np.float32)
                for i in range(len(df)):
                    self._y[i, : pawpularity[i]] = 1
            else:
                if over_100_or_not:
                    self._y = (pawpularity == 100).astype(np.float32)
                else:
                    self._y = pawpularity

        self._transform = T.Resize([image_size, image_size])

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path)
        image = self._transform(image)
        table = self._table[idx, :]
        if self._y is not None:
            label = self._y[idx]
            return image, table, label
        return image, table, 0


class PetfinderWithTableDataModule(LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        table_cols,
        cfg,
    ):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self.table_cols = table_cols
        self._cfg = cfg

    def __create_dataset(self, train=True):
        return (
            PetfinderWithTableDataset(
                self._train_df,
                self.table_cols,
                self._cfg.transform.image_size,
                self._cfg.model.output_dim,
                self._cfg.model.over_100_or_not,
            )
            if train
            else PetfinderWithTableDataset(
                self._val_df,
                self.table_cols,
                self._cfg.transform.image_size,
                self._cfg.model.output_dim,
                self._cfg.model.over_100_or_not,
            )
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._cfg.val_loader)


class PetfinderWithTableInferenceDataModule(LightningDataModule):
    def __init__(
        self,
        test_df,
        table_cols,
        cfg,
    ):
        super().__init__()
        self._test_df = test_df
        self.table_cols = table_cols

        self._cfg = cfg

    def predict_dataloader(self):
        dataset = PetfinderWithTableDataset(
            self._test_df,
            self.table_cols,
            self._cfg.transform.image_size,
            self._cfg.model.output_dim,
            self._cfg.model.over_100_or_not,
        )
        return DataLoader(dataset, **self._cfg.val_loader)


class PetfinderAdoptionSpeedDataset(Dataset):
    def __init__(self, df, image_size=224):
        self._X = df["filepath"].values
        self._y = None
        if "AdoptionSpeed" in df.keys():
            adoption_speed = df["AdoptionSpeed"].values

            self._y = np.zeros((len(df), 4)).astype(np.float32)
            for i in range(len(df)):
                self._y[i, : adoption_speed[i]] = 1

        self._transform = T.Resize([image_size, image_size])

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path)
        image = self._transform(image)
        if self._y is not None:
            label = self._y[idx]
            return image, label
        return image, 0


class PetfinderAdoptionSpeedDataModule(LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        cfg,
    ):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._cfg = cfg

    def __create_dataset(self, train=True):
        return (
            PetfinderAdoptionSpeedDataset(
                self._train_df,
                self._cfg.transform.image_size,
            )
            if train
            else PetfinderAdoptionSpeedDataset(
                self._val_df,
                self._cfg.transform.image_size,
            )
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._cfg.val_loader)


class PetfinderAdoptionSpeedInferenceDataModule(LightningDataModule):
    def __init__(
        self,
        test_df,
        cfg,
    ):
        super().__init__()
        self._test_df = test_df
        self._cfg = cfg

    def predict_dataloader(self):
        dataset = PetfinderAdoptionSpeedDataset(
            self._test_df,
            self._cfg.transform.image_size,
        )
        return DataLoader(dataset, **self._cfg.val_loader)


class PetfinderBinPawpularityDataset(Dataset):
    def __init__(self, df, image_size=224, num_bins=14):
        self._X = df["filepath"].values
        self._y = None
        if "bins_paw" in df.keys():
            adoption_speed = df["bins_paw"].values

            self._y = np.zeros((len(df), num_bins)).astype(np.float32)
            for i in range(len(df)):
                self._y[i, : adoption_speed[i]] = 1

        self._transform = T.Resize([image_size, image_size])

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path)
        image = self._transform(image)
        if self._y is not None:
            label = self._y[idx]
            return image, label
        return image, 0


class PetfinderBinPawpularityDataModule(LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        cfg,
    ):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._cfg = cfg

    def __create_dataset(self, train=True):
        return (
            PetfinderBinPawpularityDataset(
                self._train_df,
                self._cfg.transform.image_size,
                self._cfg.model.output_dim,
            )
            if train
            else PetfinderBinPawpularityDataset(
                self._val_df,
                self._cfg.transform.image_size,
                self._cfg.model.output_dim,
            )
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._cfg.val_loader)


class PetfinderBinPawpularityInferenceDataModule(LightningDataModule):
    def __init__(
        self,
        test_df,
        cfg,
    ):
        super().__init__()
        self._test_df = test_df
        self._cfg = cfg

    def predict_dataloader(self):
        dataset = PetfinderBinPawpularityDataset(
            self._test_df,
            self._cfg.transform.image_size,
        )
        return DataLoader(dataset, **self._cfg.val_loader)


class PetfinderMaturitySizeDataset(Dataset):
    def __init__(self, df, image_size=224):
        self._X = df["filepath"].values
        self._y = None
        if "MaturitySize" in df.keys():
            maturity_size = df["MaturitySize"].values

            self._y = np.zeros((len(df), 3)).astype(np.float32)
            for i in range(len(df)):
                self._y[i, : maturity_size[i]] = 1

        self._transform = T.Resize([image_size, image_size])

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path)
        image = self._transform(image)
        if self._y is not None:
            label = self._y[idx]
            return image, label
        return image, 0


class PetfinderMaturitySizeDataModule(LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        cfg,
    ):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._cfg = cfg

    def __create_dataset(self, train=True):
        return (
            PetfinderMaturitySizeDataset(
                self._train_df,
                self._cfg.transform.image_size,
            )
            if train
            else PetfinderMaturitySizeDataset(
                self._val_df,
                self._cfg.transform.image_size,
            )
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._cfg.val_loader)


class PetfinderMaturitySizeInferenceDataModule(LightningDataModule):
    def __init__(
        self,
        test_df,
        cfg,
    ):
        super().__init__()
        self._test_df = test_df
        self._cfg = cfg

    def predict_dataloader(self):
        dataset = PetfinderMaturitySizeDataset(
            self._test_df,
            self._cfg.transform.image_size,
        )
        return DataLoader(dataset, **self._cfg.val_loader)


class Petfinder1kFeatsDataset(Dataset):
    def __init__(self, df, model_config):
        self._X = df["filepath"].values
        self._transform = create_transform(**model_config)

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        # image = read_image(image_path, mode=ImageReadMode.RGB)
        image = Image.open(image_path).convert("RGB")
        image = self._transform(image)

        return image, 0


class Petfinder1kFeatsInferenceDataModule(LightningDataModule):
    def __init__(self, test_df, cfg, model_config):
        super().__init__()
        self._test_df = test_df
        self._cfg = cfg
        self._model_config = model_config

    def predict_dataloader(self):
        dataset = Petfinder1kFeatsDataset(
            self._test_df,
            self._model_config,
        )
        return DataLoader(dataset, **self._cfg.val_loader)
