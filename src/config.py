from pathlib import Path

# === Ścieżki projektu ===
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MLRUNS_DIR = ROOT_DIR / "mlruns"

# === Dane ===
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
NUM_WORKERS = 0

# === Trening ===
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30

# === Model ===
INPUT_DIM = 30
HIDDEN_DIM = 64
OUTPUT_DIM = 2
DROPOUT = 0.2

# === MLflow ===
EXPERIMENT_NAME = "mlops_project1_breast_cancer"
RUN_NAME = "baseline_mlp"
MLFLOW_TRACKING_URI = f"file:{MLRUNS_DIR.as_posix()}"

# === Optuna ===
N_TRIALS = 10
OPTUNA_STUDY_NAME = "breast_cancer_optuna_study"

# === Sprzęt ===
ACCELERATOR = "auto"
DEVICES = 1