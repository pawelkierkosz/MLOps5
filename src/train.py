import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from data_module import BreastCancerDataModule
from lightning_module import BreastCancerLightningModule
from config import (
    BATCH_SIZE,
    MAX_EPOCHS,
    ACCELERATOR,
    DEVICES,
    EXPERIMENT_NAME,
    RUN_NAME,
    MLFLOW_TRACKING_URI,
)


def main():
    data_module = BreastCancerDataModule(batch_size=BATCH_SIZE)
    model = BreastCancerLightningModule()

    mlflow_logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        run_name=RUN_NAME,
        tracking_uri=MLFLOW_TRACKING_URI
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-model-{epoch:02d}-{val_loss:.4f}"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5
    )

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        log_every_n_steps=1,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        enable_progress_bar=True
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()