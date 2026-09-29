"""
Treino final do ResNet-18 (Base) com hold-out simples (treino / validação / teste), SEM k-fold,
e exportação dos pesos para o script de inferência (resnet18_inference.py).

Configuração do modelo = a do "3. ResNet-18 (Base)" do model_comparison.py:
  sem data augmentation, Adam (lr=1e-3), ReduceLROnPlateau, loss ponderada por classe,
  checkpoint escolhido pelo Macro-F1 de validação.

Saídas:
  ../../weights/resnet18_final.pth        state_dict (torch.load + load_state_dict)
  ../../weights/resnet18_final_meta.json  ordem das classes, pré-processamento, métricas de teste
  ../../results/final_test_*              métricas, matriz de confusão, predições por imagem, split usado
"""
import copy
import json
import random
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# =========================================================
# CONFIGURAÇÕES
# =========================================================
SEED = 42

SPLITS_CSV_PATH = "../../data/k-fold_split/typography_dataset_splits.csv"
RESULTS_DIR = Path("../../results")
WEIGHTS_DIR = Path("../../weights")
WEIGHTS_NAME = "resnet18_final"

EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# True  -> hold-out AGRUPADO por gaveta (drawer_id): a mesma gaveta nunca aparece em dois conjuntos.
# False -> hold-out aleatório por imagem (estratificado por classe). Deixa letras da mesma gaveta em
#          treino e teste ao mesmo tempo (vazamento) e infla as métricas.
SPLIT_BY_DRAWER = True
N_TEST_SPLITS = 7      # 1/7  ~ 14% das imagens para teste
N_VAL_SPLITS = 6       # 1/6 do restante ~ 14% para validação  ->  treino ~ 71%
MAX_SPLIT_ATTEMPTS = 500
MIN_PER_CLASS_WARN = 5  # avisa se alguma classe tiver menos que isso em validação/teste

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executando na plataforma: {device}")


# =========================================================
# 1. DATASET E TRANSFORM
# =========================================================
class TypographyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, 'image_path']
        label = self.dataframe.loc[idx, 'label_encoded']

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# Deve ser idêntico ao pré-processamento do resnet18_inference.py
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])


# =========================================================
# 2. SPLIT TREINO / VALIDAÇÃO / TESTE
# =========================================================
def make_grouped_holdout_split(df, n_test_splits=N_TEST_SPLITS, n_val_splits=N_VAL_SPLITS,
                               seed=SEED, max_attempts=MAX_SPLIT_ATTEMPTS):
    """
    Hold-out estratificado E agrupado por gaveta.

    1) Um fold do StratifiedGroupKFold(n_test_splits) vira o TESTE.
    2) Do restante, um fold do StratifiedGroupKFold(n_val_splits) vira a VALIDAÇÃO.
    3) O que sobra é o TREINO.

    Como o StratifiedGroupKFold é uma heurística, tentamos várias sementes, exigimos pelo menos
    1 imagem de cada classe em cada conjunto e ficamos com a divisão mais parecida com a
    distribuição global de classes e com os tamanhos-alvo.
    """
    if "drawer_id" not in df.columns:
        raise ValueError("O CSV precisa da coluna 'drawer_id' para o split por gaveta. "
                         "Gere-o com dataset_split_kfold.py ou use SPLIT_BY_DRAWER = False.")

    df = df.reset_index(drop=True)
    class_names = sorted(df["label"].unique())
    n_total = len(df)
    class_totals = df["label"].value_counts().reindex(class_names)

    target_frac = {"test": 1.0 / n_test_splits}
    target_frac["val"] = (1.0 - target_frac["test"]) / n_val_splits
    target_frac["train"] = 1.0 - target_frac["test"] - target_frac["val"]

    # Duas categorias de candidatas: as que têm >= MIN_PER_CLASS_WARN imagens de cada classe em
    # validação e teste (preferidas, pois o recall de classes raras fica menos ruidoso) e as demais.
    best = {True: (None, np.inf, None), False: (None, np.inf, None)}
    n_valid = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # avisos de "classe rara" do sklearn

        for attempt in range(max_attempts):
            s = seed + attempt

            trainval_idx, test_idx = next(StratifiedGroupKFold(
                n_splits=n_test_splits, shuffle=True, random_state=s
            ).split(df, df["label"], df["drawer_id"]))

            trainval = df.iloc[trainval_idx]
            tr_rel, va_rel = next(StratifiedGroupKFold(
                n_splits=n_val_splits, shuffle=True, random_state=s
            ).split(trainval, trainval["label"], trainval["drawer_id"]))

            idx = {"train": trainval_idx[tr_rel], "val": trainval_idx[va_rel], "test": test_idx}

            # contagens (conjuntos x classes)
            counts = pd.DataFrame(
                {name: df.iloc[i]["label"].value_counts() for name, i in idx.items()}
            ).reindex(class_names).fillna(0).T
            counts = counts.loc[["train", "val", "test"]]

            if counts.values.min() < 1:
                continue
            n_valid += 1

            frac = counts.sum(axis=1) / n_total
            expected = np.outer(frac.values, class_totals.values)
            mix_score = (((counts.values - expected) / expected) ** 2).sum()
            size_score = sum(((frac[name] - target_frac[name]) / target_frac[name]) ** 2
                             for name in idx)
            score = mix_score + size_score

            robust = bool(counts.loc[["val", "test"]].values.min() >= MIN_PER_CLASS_WARN)
            if score < best[robust][1]:
                best[robust] = (idx, score, s)

    best_idx, best_score, best_seed = best[True] if best[True][0] is not None else best[False]

    if best_idx is None:
        raise RuntimeError(
            f"Nenhuma das {max_attempts} tentativas deixou todas as classes presentes em treino, "
            f"validação e teste. Reduza N_TEST_SPLITS/N_VAL_SPLITS ou aumente MAX_SPLIT_ATTEMPTS."
        )

    print(f"Split por gaveta: {n_valid}/{max_attempts} tentativas válidas; "
          f"semente escolhida = {best_seed} (score = {best_score:.4f})")
    if best[True][0] is None:
        print(f"AVISO: nenhuma divisão tem >= {MIN_PER_CLASS_WARN} imagens de cada classe em validação e "
              f"teste; usando a melhor disponível. Considere reduzir N_TEST_SPLITS/N_VAL_SPLITS.")

    parts = {name: df.iloc[i].reset_index(drop=True) for name, i in best_idx.items()}

    # Segurança: nenhuma gaveta em mais de um conjunto
    drawers = {name: set(p["drawer_id"]) for name, p in parts.items()}
    assert not (drawers["train"] & drawers["val"]), "Vazamento: gaveta em treino e validação!"
    assert not (drawers["train"] & drawers["test"]), "Vazamento: gaveta em treino e teste!"
    assert not (drawers["val"] & drawers["test"]), "Vazamento: gaveta em validação e teste!"

    return parts["train"], parts["val"], parts["test"]


def make_random_holdout_split(df, test_size=0.15, val_size=0.15, seed=SEED):
    """Hold-out aleatório estratificado por classe (NÃO agrupa por gaveta)."""
    print("AVISO: split aleatório por imagem — letras da mesma gaveta podem cair em treino e teste, "
          "o que infla as métricas.")

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label_encoded"], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size / (1.0 - test_size),
        stratify=train_val_df["label_encoded"], random_state=seed
    )
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def report_split(train_df, val_df, test_df, class_names):
    parts = {"train": train_df, "val": val_df, "test": test_df}
    total = sum(len(p) for p in parts.values())

    print("\nImagens por classe em cada conjunto:")
    table = pd.DataFrame({name: p["label"].value_counts() for name, p in parts.items()}) \
        .reindex(class_names).fillna(0).astype(int).T
    table["total"] = table.sum(axis=1)
    table["% do total"] = (table["total"] / total * 100).round(1)
    print(table.to_string())

    if "drawer_id" in train_df.columns:
        print("\nGavetas por conjunto: " +
              " | ".join(f"{name}: {p['drawer_id'].nunique()}" for name, p in parts.items()))

    for name in ("val", "test"):
        low = table.loc[name, class_names]
        low = low[low < MIN_PER_CLASS_WARN]
        if not low.empty:
            print(f"AVISO: {name} tem poucas imagens de {low.to_dict()} — "
                  f"o recall dessas classes será muito ruidoso.")


# =========================================================
# 3. TREINO E AVALIAÇÃO
# =========================================================
def compute_class_weights(train_df, num_classes, power=1.0):
    """Pesos inversamente proporcionais à frequência: w_c = (N / (K * n_c)) ** power."""
    counts = np.bincount(train_df["label_encoded"].values, minlength=num_classes).astype(float)
    weights = (counts.sum() / (num_classes * np.maximum(counts, 1.0))) ** power
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, criterion, num_classes):
    """Retorna (loss médio, acurácia, macro-F1) em um DataLoader."""
    model.eval()

    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    loss = running_loss / len(all_labels)
    accuracy = float((all_preds == all_labels).mean())
    macro_f1 = f1_score(all_labels, all_preds, labels=np.arange(num_classes),
                        average="macro", zero_division=0)
    return loss, accuracy, macro_f1


def train_resnet18(train_loader, val_loader, num_classes, class_weights,
                   epochs=EPOCHS, patience=PATIENCE):
    print(f"\nTreinando ResNet-18 (Final) por até {epochs} épocas...")

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=2)

    print(f"Pesos de classe na loss: {[round(w, 3) for w in class_weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # treino: ponderada
    eval_criterion = nn.CrossEntropyLoss()                            # validação: só monitoramento

    best_score = -float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0

    epoch_iterator = tqdm(range(epochs), desc="Épocas (ResNet-18 Final)", unit="época")
    train_start = time.perf_counter()

    for epoch in epoch_iterator:
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, eval_criterion, num_classes)
        scheduler.step(val_f1)  # checkpoint, scheduler e early stopping guiados pelo Macro-F1

        if val_f1 >= best_score:
            best_score = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_iterator.set_postfix({
            "Train Loss": f"{epoch_loss:.4f}",
            "Val Loss": f"{val_loss:.4f}",
            "Val Acc": f"{val_acc:.4f}",
            "Val F1": f"{val_f1:.4f}",
            "Best F1": f"{best_score:.4f}",
            "LR": f"{optimizer.param_groups[0]['lr']:.1e}"
        })

        if epochs_without_improvement >= patience:
            print(f"\nEarly Stopping na época {epoch + 1}")
            break

    model.load_state_dict(best_weights)
    train_time = time.perf_counter() - train_start
    print(f"\nMelhor Macro-F1 de validação: {best_score:.4f} (época {best_epoch})")
    print(f"Tempo de treinamento: {train_time:.2f} s")
    return model, train_time, best_epoch, best_score


def predict_dl_model(model, test_loader):
    """Retorna (predições, matriz de probabilidades N x K)."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs.to(device))
            all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    return probs.argmax(axis=1), probs


# =========================================================
# 4. RELATÓRIOS DO TESTE
# =========================================================
def save_confusion_figure(cm, cm_norm, class_names, path, title):
    """Matriz normalizada por linha (recall por classe), anotada com % e contagem."""
    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_norm[i, j] * 100:.1f}%\n(n={cm[i, j]})"

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", vmin=0.0, vmax=1.0,
                xticklabels=class_names, yticklabels=class_names, cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predição")
    ax.set_ylabel("Classe verdadeira")
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 5. PIPELINE PRINCIPAL
# =========================================================
def run_final_test(df):
    print("\n" + "=" * 70)
    print("TREINAMENTO FINAL — RESNET-18 (BASE) COM HOLD-OUT (SEM K-FOLD)")
    print("=" * 70)

    # A ordem das classes vem do LabelEncoder (alfabética) e é salva junto com os pesos.
    le = LabelEncoder()
    df = df.copy()
    df["label_encoded"] = le.fit_transform(df["label"])
    class_names = [str(c) for c in le.classes_]
    num_classes = len(class_names)

    # ---------------- Split ----------------
    if SPLIT_BY_DRAWER:
        train_df, val_df, test_df = make_grouped_holdout_split(df)
    else:
        train_df, val_df, test_df = make_random_holdout_split(df)

    report_split(train_df, val_df, test_df, class_names)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    split_df = pd.concat([
        train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test")
    ], ignore_index=True)
    split_df.drop(columns=["label_encoded"]).to_csv(RESULTS_DIR / "final_test_split.csv", index=False)

    # ---------------- DataLoaders ----------------
    class_weights = compute_class_weights(train_df, num_classes)

    g = torch.Generator()
    g.manual_seed(SEED)

    loader_train = DataLoader(TypographyDataset(train_df, transform=base_transform),
                              batch_size=BATCH_SIZE, shuffle=True, generator=g)
    loader_val = DataLoader(TypographyDataset(val_df, transform=base_transform),
                            batch_size=BATCH_SIZE, shuffle=False)
    loader_test = DataLoader(TypographyDataset(test_df, transform=base_transform),
                             batch_size=BATCH_SIZE, shuffle=False)

    # ---------------- Treino ----------------
    model, train_time, best_epoch, best_val_f1 = train_resnet18(
        loader_train, loader_val, num_classes, class_weights
    )

    # ---------------- Exporta os pesos (antes de olhar o teste) ----------------
    weights_path = WEIGHTS_DIR / f"{WEIGHTS_NAME}.pth"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, weights_path)
    print(f"\nPesos salvos em: {weights_path.resolve()}")

    # ---------------- Teste (uma única vez) ----------------
    y_true = test_df["label_encoded"].values

    start = time.perf_counter()
    y_pred, probs = predict_dl_model(model, loader_test)   # inclui leitura do disco + transformações
    inference_time = time.perf_counter() - start

    report = classification_report(
        y_true, y_pred, labels=np.arange(num_classes), target_names=class_names,
        output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose().round(4)
    report_df.to_csv(RESULTS_DIR / "final_test_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    cm_norm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes), normalize="true")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        RESULTS_DIR / "final_test_confusion_counts.csv")
    save_confusion_figure(cm, cm_norm, class_names, RESULTS_DIR / "final_test_confusion.pdf",
                          "ResNet-18 (Final) — conjunto de teste")

    # Predições por imagem (útil para analisar erros por gaveta)
    pred_df = pd.DataFrame({
        "image_path": test_df["image_path"],
        "drawer_id": test_df["drawer_id"] if "drawer_id" in test_df.columns else "",
        "label": test_df["label"],
        "predicted": [class_names[i] for i in y_pred],
        "confidence": probs.max(axis=1),
    })
    for i, c in enumerate(class_names):
        pred_df[f"prob_{c}"] = probs[:, i]
    pred_df.to_csv(RESULTS_DIR / "final_test_predictions.csv", index=False)

    fantasy_recall = report["fantasia"]["recall"] if "fantasia" in report else float("nan")
    summary = pd.DataFrame([{
        "Model": "ResNet-18 (Final)",
        "Accuracy": report["accuracy"],
        "Precision Macro": report["macro avg"]["precision"],
        "Recall Macro": report["macro avg"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"],
        "Weighted F1": report["weighted avg"]["f1-score"],
        "Fantasy Recall": fantasy_recall,
        "Best Val Macro F1": best_val_f1,
        "Best Epoch": best_epoch,
        "Train Time (s)": train_time,
        "Inference Time (s)": inference_time,
        "Inference ms/img": inference_time / len(y_true) * 1000,
        "N Test": len(y_true),
    }])
    summary.to_csv(RESULTS_DIR / "final_test_benchmark_results.csv", index=False)

    print("\n" + "=" * 70)
    print("RESULTADO NO CONJUNTO DE TESTE")
    print("=" * 70)
    print(summary.round(4).T.to_string(header=False))
    print("\nClassification report:")
    print(report_df)

    # ---------------- Metadados para o script de inferência ----------------
    meta = {
        "architecture": "resnet18",
        "num_classes": num_classes,
        "class_names": class_names,
        "input_size": [224, 224],
        "normalize_mean": NORM_MEAN,
        "normalize_std": NORM_STD,
        "input_mode": "grayscale_replicated_to_rgb",
        "weights_file": weights_path.name,
        "seed": SEED,
        "best_epoch": int(best_epoch),
        "split": {
            "by_drawer": SPLIT_BY_DRAWER,
            "train": len(train_df), "val": len(val_df), "test": len(test_df),
        },
        "test_metrics": {
            "accuracy": float(report["accuracy"]),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "fantasy_recall": float(fantasy_recall),
        },
    }
    meta_path = WEIGHTS_DIR / f"{WEIGHTS_NAME}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Metadados salvos em: {meta_path.resolve()}")

    print("\nArquivos gerados em", RESULTS_DIR.resolve())
    for name in ["final_test_split.csv", "final_test_classification_report.csv",
                 "final_test_confusion_counts.csv", "final_test_confusion.pdf",
                 "final_test_predictions.csv", "final_test_benchmark_results.csv"]:
        print(f"  • {name}")


if __name__ == "__main__":
    df = pd.read_csv(SPLITS_CSV_PATH)
    run_final_test(df)
