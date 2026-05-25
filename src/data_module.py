import lightning as L
from torch.utils.data import DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset import TabularDataset
from config import RANDOM_STATE, TEST_SIZE, VAL_SIZE, BATCH_SIZE, NUM_WORKERS


class BreastCancerDataModule(L.LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.scaler = StandardScaler()

    def setup(self, stage=None):
        data = load_breast_cancer()
        X = data.data
        y = data.target

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=VAL_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_train_val
        )

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        self.train_dataset = TabularDataset(X_train, y_train)
        self.val_dataset = TabularDataset(X_val, y_val)
        self.test_dataset = TabularDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )