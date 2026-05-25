import copy
import time
from pathlib import Path
import tempfile

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from data_module import BreastCancerDataModule
from lightning_module import BreastCancerLightningModule
from model import MLPClassifier
from config import BATCH_SIZE, ROOT_DIR, INPUT_DIM, OUTPUT_DIM, DROPOUT


DEVICE = torch.device("cpu")
WARMUP_ITERATIONS = 20
MEASURE_ITERATIONS = 300
UNSTRUCTURED_PRUNING_AMOUNT = 0.5
STRUCTURED_HIDDEN_DIM = 32


def find_best_checkpoint():
    checkpoints_dir = ROOT_DIR / "lightning_logs"

    checkpoint_paths = list(checkpoints_dir.glob("**/checkpoints/*.ckpt"))

    if not checkpoint_paths:
        raise FileNotFoundError(
            "Nie znaleziono checkpointu. Najpierw uruchom: python src/train.py"
        )

    checkpoint_paths = sorted(
        checkpoint_paths,
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    return checkpoint_paths[0]


def load_trained_model(checkpoint_path):
    lightning_model = BreastCancerLightningModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path)
    )

    model = copy.deepcopy(lightning_model.model)
    model.to(DEVICE)
    model.eval()

    return model


def get_test_loader():
    data_module = BreastCancerDataModule(batch_size=BATCH_SIZE)
    data_module.setup()
    return data_module.test_dataloader()


def evaluate_model(model, test_loader):
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.inference_mode():
        for X, y in test_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    f1 = f1_score(all_targets, all_predictions)
    accuracy = accuracy_score(all_targets, all_predictions)

    return f1, accuracy


def measure_inference_time(model, test_loader):
    model.eval()

    batches = list(test_loader)

    with torch.inference_mode():
        for _ in range(WARMUP_ITERATIONS):
            for X, _ in batches:
                X = X.to(DEVICE)
                _ = model(X)

    start_time = time.perf_counter()

    with torch.inference_mode():
        for _ in range(MEASURE_ITERATIONS):
            for X, _ in batches:
                X = X.to(DEVICE)
                _ = model(X)

    end_time = time.perf_counter()

    total_batches = MEASURE_ITERATIONS * len(batches)
    average_time_per_batch = (end_time - start_time) / total_batches

    return average_time_per_batch


def get_serialized_model_size_kb(model):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        torch.save(model.state_dict(), temp_path)
        size_kb = temp_path.stat().st_size / 1024

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    return size_kb


def get_theoretical_size_kb(model, bits_per_parameter):
    total_parameters = 0

    for parameter in model.parameters():
        total_parameters += parameter.numel()

    size_kb = total_parameters * bits_per_parameter / 8 / 1024

    return size_kb


def count_nonzero_parameters(model):
    nonzero_params = 0

    for parameter in model.parameters():
        nonzero_params += torch.count_nonzero(parameter).item()

    return nonzero_params


def count_all_parameters(model):
    total_params = 0

    for parameter in model.parameters():
        total_params += parameter.numel()

    return total_params


def fake_quantize_tensor(tensor, bits):
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1

    max_abs = tensor.abs().max()

    if max_abs == 0:
        return tensor.clone()

    scale = max_abs / qmax

    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, qmin, qmax)

    dequantized = quantized * scale

    return dequantized


def create_int16_model(base_model):
    model = copy.deepcopy(base_model)

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = fake_quantize_tensor(module.weight.data, bits=16)

                if module.bias is not None:
                    module.bias.data = fake_quantize_tensor(module.bias.data, bits=16)

    model.eval()
    return model


def create_int8_dynamic_model(base_model):
    model = copy.deepcopy(base_model)
    model.eval()

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    quantized_model.eval()
    return quantized_model


def create_unstructured_pruned_model(base_model, amount=UNSTRUCTURED_PRUNING_AMOUNT):
    model = copy.deepcopy(base_model)

    parameters_to_prune = []

    for module in model.modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )

    for module, parameter_name in parameters_to_prune:
        prune.remove(module, parameter_name)

    model.eval()
    return model


def create_structured_pruned_model(base_model):
    old_layers = []

    for module in base_model.network:
        if isinstance(module, nn.Linear):
            old_layers.append(module)

    old_l1 = old_layers[0]
    old_l2 = old_layers[1]
    old_l3 = old_layers[2]

    old_hidden_dim = old_l1.out_features
    old_second_hidden_dim = old_l2.out_features

    new_hidden_dim = STRUCTURED_HIDDEN_DIM
    new_second_hidden_dim = STRUCTURED_HIDDEN_DIM // 2

    if new_hidden_dim >= old_hidden_dim:
        raise ValueError("STRUCTURED_HIDDEN_DIM musi być mniejsze od starego hidden_dim.")

    if new_second_hidden_dim >= old_second_hidden_dim:
        raise ValueError("Druga warstwa po structured pruning musi być mniejsza od starej.")

    first_layer_scores = old_l1.weight.data.abs().sum(dim=1)
    selected_first_neurons = torch.topk(
        first_layer_scores,
        k=new_hidden_dim
    ).indices.sort().values

    second_layer_scores = old_l2.weight.data.abs().sum(dim=1)
    selected_second_neurons = torch.topk(
        second_layer_scores,
        k=new_second_hidden_dim
    ).indices.sort().values

    new_model = MLPClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=new_hidden_dim,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT
    )

    new_layers = []

    for module in new_model.network:
        if isinstance(module, nn.Linear):
            new_layers.append(module)

    new_l1 = new_layers[0]
    new_l2 = new_layers[1]
    new_l3 = new_layers[2]

    with torch.no_grad():
        new_l1.weight.data = old_l1.weight.data[selected_first_neurons, :].clone()
        new_l1.bias.data = old_l1.bias.data[selected_first_neurons].clone()

        new_l2.weight.data = old_l2.weight.data[
            selected_second_neurons, :
        ][:, selected_first_neurons].clone()
        new_l2.bias.data = old_l2.bias.data[selected_second_neurons].clone()

        new_l3.weight.data = old_l3.weight.data[:, selected_second_neurons].clone()
        new_l3.bias.data = old_l3.bias.data.clone()

    new_model.to(DEVICE)
    new_model.eval()

    return new_model


def run_single_evaluation(model, test_loader, model_size_kb=None):
    f1, accuracy = evaluate_model(model, test_loader)
    inference_time = measure_inference_time(model, test_loader)

    if model_size_kb is None:
        model_size_kb = get_serialized_model_size_kb(model)

    total_params = count_all_parameters(model)
    nonzero_params = count_nonzero_parameters(model)

    return {
        "F1 score": f1,
        "Accuracy": accuracy,
        "Inference time [s/batch]": inference_time,
        "Model size [KB]": model_size_kb,
        "Total params": total_params,
        "Non-zero params": nonzero_params
    }


def run_quantization_experiment(base_model, test_loader):
    float32_model = copy.deepcopy(base_model)
    int16_model = create_int16_model(base_model)
    int8_model = create_int8_dynamic_model(base_model)

    quantization_results = []

    float32_result = run_single_evaluation(
        float32_model,
        test_loader,
        model_size_kb=get_theoretical_size_kb(float32_model, bits_per_parameter=32)
    )
    float32_result["Precision"] = "float32"
    quantization_results.append(float32_result)

    int16_result = run_single_evaluation(
        int16_model,
        test_loader,
        model_size_kb=get_theoretical_size_kb(int16_model, bits_per_parameter=16)
    )
    int16_result["Precision"] = "int16 simulated"
    quantization_results.append(int16_result)

    int8_result = run_single_evaluation(
        int8_model,
        test_loader,
        model_size_kb=get_serialized_model_size_kb(int8_model)
    )
    int8_result["Precision"] = "int8 dynamic"
    quantization_results.append(int8_result)

    table = pd.DataFrame(quantization_results)

    table = table[
        [
            "Precision",
            "F1 score",
            "Accuracy",
            "Inference time [s/batch]",
            "Model size [KB]"
        ]
    ]

    return table


def run_pruning_experiment(base_model, test_loader):
    baseline_model = copy.deepcopy(base_model)
    unstructured_model = create_unstructured_pruned_model(base_model)
    structured_model = create_structured_pruned_model(base_model)

    pruning_results = []

    baseline_result = run_single_evaluation(baseline_model, test_loader)
    baseline_result["Variant"] = "Baseline"
    pruning_results.append(baseline_result)

    unstructured_result = run_single_evaluation(unstructured_model, test_loader)
    unstructured_result["Variant"] = "Unstructured pruning 50%"
    pruning_results.append(unstructured_result)

    structured_result = run_single_evaluation(structured_model, test_loader)
    structured_result["Variant"] = "Structured pruning"
    pruning_results.append(structured_result)

    table = pd.DataFrame(pruning_results)

    table = table[
        [
            "Variant",
            "F1 score",
            "Accuracy",
            "Inference time [s/batch]",
            "Total params",
            "Non-zero params"
        ]
    ]

    return table


def save_results_to_markdown(quantization_table, pruning_table):
    output_path = ROOT_DIR / "HOMEWORK2_RESULTS.md"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("# Homework 2 - Inference Optimization Results\n\n")

        file.write("## Quantization\n\n")
        file.write(quantization_table.to_markdown(index=False))
        file.write("\n\n")

        file.write("## Pruning\n\n")
        file.write(pruning_table.to_markdown(index=False))
        file.write("\n\n")

        file.write("## Short interpretation\n\n")
        file.write(
            "Unstructured pruning zeros individual weights, but the shapes of the weight matrices stay the same. "
            "Therefore, the standard dense backend still performs almost the same number of operations.\n\n"
        )
        file.write(
            "Structured pruning removes whole neurons and reduces the actual dimensions of linear layers. "
            "This can reduce the number of operations and may improve inference speed.\n"
        )

    print(f"\nWyniki zapisano do pliku: {output_path}")


def main():
    checkpoint_path = find_best_checkpoint()

    print("Używany checkpoint:")
    print(checkpoint_path)

    test_loader = get_test_loader()
    base_model = load_trained_model(checkpoint_path)

    print("\nUruchamiam eksperyment kwantyzacji...")
    quantization_table = run_quantization_experiment(base_model, test_loader)

    print("\n=== QUANTIZATION RESULTS ===")
    print(quantization_table.to_string(index=False))

    print("\nUruchamiam eksperyment pruningu...")
    pruning_table = run_pruning_experiment(base_model, test_loader)

    print("\n=== PRUNING RESULTS ===")
    print(pruning_table.to_string(index=False))

    save_results_to_markdown(quantization_table, pruning_table)


if __name__ == "__main__":
    main()