import optuna
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from data_module import BreastCancerDataModule
from lightning_module import BreastCancerLightningModule
from config import (
    INPUT_DIM,
    OUTPUT_DIM,
    MAX_EPOCHS,
    ACCELERATOR,
    DEVICES,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    N_TRIALS,
    OPTUNA_STUDY_NAME,
)


def objective(trial):
    # === Hiperparametry do strojenia ===
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # === Dane ===
    data_module = BreastCancerDataModule(batch_size=batch_size)

    # === Model ===
    model = BreastCancerLightningModule(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        output_dim=OUTPUT_DIM,
        dropout=dropout,
        learning_rate=learning_rate
    )

    # === Logger MLflow ===
    mlflow_logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        run_name=f"optuna_trial_{trial.number}",
        tracking_uri=MLFLOW_TRACKING_URI
    )

    # Logowanie hiperparametrów triala do MLflow
    mlflow_logger.log_hyperparams({
        "trial_number": trial.number,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "batch_size": batch_size,
    })

    # === Callbacki ===
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=f"trial-{trial.number}" + "-{epoch:02d}-{val_loss:.4f}"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5
    )

    # === Trainer ===
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=1
    )

    # === Trening ===
    trainer.fit(model, datamodule=data_module)

    # Bierzemy końcową metrykę walidacyjną
    val_acc = trainer.callback_metrics.get("val_acc")

    if val_acc is None:
        raise ValueError("Nie udało się odczytać val_acc z callback_metrics.")

    return val_acc.item()


def main():
    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        direction="maximize"
    )

    study.optimize(objective, n_trials=N_TRIALS)

    print("\n=== Najlepszy trial ===")
    print("Wartość (best val_acc):", study.best_value)
    print("Najlepsze hiperparametry:")

    for key, value in study.best_params.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()