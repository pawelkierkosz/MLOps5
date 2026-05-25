````markdown
# Homework 2 - Inference Optimization

Projekt jest kontynuacją pierwszego zadania z przedmiotu **Narzędzia Uczenia Maszynowego / MLOps**.

W pierwszym projekcie został przygotowany model klasyfikacyjny w **PyTorch Lightning** dla danych tabelarycznych.  
W tym zadaniu wykorzystano ten sam wytrenowany model i sprawdzono dwie techniki optymalizacji inferencji:

- **quantization**
- **pruning**

Celem było porównanie jakości modelu, czasu inferencji oraz rozmiaru/liczby parametrów po zastosowaniu optymalizacji.

---

## Model bazowy

Model pochodzi z Homework 1.

Wykorzystany model:

- typ modelu: MLP
- typ danych: dane tabelaryczne
- dataset: Breast Cancer Wisconsin Dataset
- zadanie: klasyfikacja binarna
- framework: PyTorch + PyTorch Lightning

Model został załadowany z najlepszego checkpointu zapisanego podczas treningu:

```text
lightning_logs/version_1/checkpoints/best-model-epoch=15-val_loss=0.0638.ckpt
````

---

## Dataset

W projekcie wykorzystano dataset `load_breast_cancer()` z biblioteki `scikit-learn`.

Charakterystyka danych:

* dane tabelaryczne
* 30 cech wejściowych
* klasyfikacja binarna
* ten sam zbiór testowy dla wszystkich wariantów modelu

Dzięki użyciu tego samego zbioru testowego wyniki poszczególnych wersji modelu są porównywalne.

---

## Co zostało wykonane?

W zadaniu sprawdzono dwie grupy metod optymalizacji inferencji:

1. Quantization
2. Pruning

Dla każdej wersji modelu zmierzono:

* F1 score
* accuracy
* czas inferencji
* rozmiar modelu albo liczbę niezerowych parametrów

Do pomiaru czasu użyto trybu ewaluacji modelu oraz wyłączono liczenie gradientów.

---

## Uruchomienie

Najpierw należy upewnić się, że istnieje checkpoint modelu z Homework 1.

Jeżeli checkpoint nie istnieje, należy uruchomić trening:

```bash
python src/train.py
```

Następnie można uruchomić eksperyment optymalizacji inferencji:

```bash
python src/inference_optimization.py
```

Po uruchomieniu skryptu wyniki są wypisywane w konsoli oraz zapisywane do pliku:

```text
HOMEWORK2_RESULTS.md
```

---

## Quantization

W tej części porównano trzy wersje modelu:

* `float32` - oryginalny model
* `int16 simulated` - symulowana kwantyzacja 16-bitowa
* `int8 dynamic` - dynamiczna kwantyzacja 8-bitowa

Celem było sprawdzenie, czy zmniejszenie precyzji wag wpływa na jakość modelu, czas inferencji i rozmiar modelu.

### Quantization results

| Precision       | F1 score | Accuracy | Inference time [s/batch] | Model size [KB] |
| :-------------- | -------: | -------: | -----------------------: | --------------: |
| float32         | 0.964539 |  0.95614 |              9.77895e-05 |         16.1328 |
| int16 simulated | 0.964539 |  0.95614 |              9.78742e-05 |         8.06641 |
| int8 dynamic    | 0.964539 |  0.95614 |              0.000414759 |         9.44434 |

### Quantization interpretation

Wyniki pokazują, że kwantyzacja nie pogorszyła jakości modelu.
Dla wszystkich trzech wersji model osiągnął taki sam wynik F1 score:

```text
F1 score = 0.964539
```

Model `int16 simulated` zmniejszył teoretyczny rozmiar modelu z około `16.13 KB` do około `8.07 KB`.

Model `int8 dynamic` również zmniejszył rozmiar modelu względem wersji `float32`, ale w tym przypadku czas inferencji był większy niż dla modelu bazowego.

Możliwe wyjaśnienie jest takie, że model MLP jest bardzo mały, więc narzut związany z dynamiczną kwantyzacją może być większy niż potencjalny zysk z mniejszej precyzji obliczeń.

Wniosek:

* kwantyzacja zmniejszyła rozmiar modelu
* jakość modelu została zachowana
* dla tego małego modelu nie uzyskano przyspieszenia inferencji

---

## Pruning

W tej części porównano trzy warianty modelu:

* `Baseline`
* `Unstructured pruning 50%`
* `Structured pruning`

Celem było sprawdzenie, czy usunięcie części wag lub neuronów wpływa na jakość modelu i czas inferencji.

### Pruning results

| Variant                  | F1 score | Accuracy | Inference time [s/batch] | Total params | Non-zero params |
| :----------------------- | -------: | -------: | -----------------------: | -----------: | --------------: |
| Baseline                 | 0.964539 |  0.95614 |               8.9893e-05 |         4130 |            4130 |
| Unstructured pruning 50% | 0.964539 |  0.95614 |              8.91325e-05 |         4130 |            2114 |
| Structured pruning       |  0.97931 | 0.973684 |              7.64848e-05 |         1554 |            1554 |

### Pruning interpretation

Unstructured pruning zmniejszył liczbę niezerowych parametrów:

```text
4130 -> 2114
```

Jednak całkowita liczba parametrów pozostała taka sama:

```text
4130
```

Oznacza to, że pojedyncze wagi zostały wyzerowane, ale kształty macierzy wag się nie zmieniły.
Standardowy dense backend nadal wykonuje operacje na macierzach o tych samych wymiarach, dlatego czas inferencji prawie się nie zmienił.

Structured pruning zmniejszył całkowitą liczbę parametrów:

```text
4130 -> 1554
```

W tym przypadku usunięto całe neurony, więc rzeczywiste wymiary warstw liniowych zostały zmniejszone.
Dzięki temu liczba wykonywanych operacji była mniejsza, a czas inferencji spadł.

W tym eksperymencie structured pruning dodatkowo poprawił wynik F1 score:

```text
0.964539 -> 0.97931
```

Może to wynikać z tego, że mniejszy model działał jak forma regularizacji.

---

## Najważniejsze wnioski

1. Kwantyzacja zmniejszyła rozmiar modelu bez utraty jakości.
2. Dynamiczna kwantyzacja int8 nie przyspieszyła inferencji w tym przypadku, prawdopodobnie dlatego, że model był bardzo mały.
3. Unstructured pruning zmniejszył liczbę niezerowych wag, ale nie zmniejszył rozmiarów macierzy.
4. Unstructured pruning nie dał realnego przyspieszenia inferencji.
5. Structured pruning zmniejszył rzeczywistą liczbę parametrów i lekko przyspieszył inferencję.
6. Structured pruning może realnie zmniejszyć liczbę operacji, ponieważ usuwa całe neurony, a nie tylko zeruje pojedyncze wagi.

---

## Struktura projektu

```text
MLOps_Project1/
│
├── data/
├── lightning_logs/
├── mlruns/
├── src/
│   ├── config.py
│   ├── data_module.py
│   ├── dataset.py
│   ├── inference_optimization.py
│   ├── lightning_module.py
│   ├── model.py
│   ├── train.py
│   ├── tune_optuna.py
│   └── utils.py
│
├── HOMEWORK2_RESULTS.md
├── README.md
└── requirements.txt
```

---

## Najważniejsze pliki

* `src/model.py` - definicja modelu MLP
* `src/lightning_module.py` - moduł PyTorch Lightning
* `src/data_module.py` - przygotowanie danych i DataLoaderów
* `src/train.py` - trening modelu bazowego
* `src/inference_optimization.py` - eksperymenty kwantyzacji i pruningu
* `HOMEWORK2_RESULTS.md` - zapisane wyniki eksperymentów
* `requirements.txt` - wymagane biblioteki

---

## Wymagane biblioteki

Projekt korzysta z następujących bibliotek:

* torch
* lightning
* torchmetrics
* scikit-learn
* numpy
* pandas
* mlflow
* optuna
* tabulate

Instalacja zależności:

```bash
pip install -r requirements.txt
```

---

## Podsumowanie

Projekt pokazuje, że techniki optymalizacji inferencji mogą zmniejszyć rozmiar modelu lub liczbę parametrów, ale nie zawsze automatycznie poprawiają czas działania.

Najważniejsza obserwacja dotyczy pruningu:

* samo wyzerowanie wag nie wystarcza, aby model był szybszy
* aby realnie zmniejszyć liczbę operacji, trzeba zmienić strukturę modelu, np. usunąć całe neurony

W tym eksperymencie najlepszy kompromis uzyskano dla structured pruning, ponieważ zmniejszył liczbę parametrów, poprawił F1 score i lekko skrócił czas inferencji.

```
```
