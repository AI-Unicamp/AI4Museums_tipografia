import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from PIL import Image
import re

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random

SEED = 42

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
        except Exception as e:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_dl_model(model, test_loader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)

def evaluate_model(model, X_test, class_names, y_test, model_type="sklearn", test_loader=None, train_time=None):
    """
    Avalia um modelo treinado.
    """

    start = time.perf_counter()

    if model_type == "sklearn":
        y_pred = model.predict(X_test)

    elif model_type == "pytorch":
        y_pred = predict_dl_model(model, test_loader)

    else:
        raise ValueError(
            "model_type deve ser 'sklearn' ou 'pytorch'."
        )

    inference_time = time.perf_counter() - start

    report = classification_report(
        y_test,
        y_pred,
	    labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    return {
        "predictions": y_pred,
        "accuracy": report["accuracy"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "fantasy_recall": report["fantasia"]["recall"],
        "train_time": train_time,
        "inference_time": inference_time,
        "classification_report": report
    }

def evaluate(model, loader, criterion):
    """
    Avalia o modelo em um DataLoader.
    Retorna:
        loss médio
        accuracy
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

    loss = running_loss / total
    accuracy = correct / total

    return loss, accuracy

def evaluate_all_models(y_true, predictions_dict, class_names, save_prefix=""):

    print("\n" + "=" * 60)
    print("GERANDO RESULTADOS COMPARATIVOS")
    print("=" * 60)

    results_dir = Path("../../results")
    results_dir.mkdir(parents=True, exist_ok=True)

    save_prefix = results_dir / save_prefix

    results = []
    for model_name, result in predictions_dict.items():

        results.append({
            "Model": model_name,
            "Accuracy": result["accuracy"],
            "Precision Macro": result["precision_macro"],
            "Recall Macro": result["recall_macro"],
            "Macro F1": result["f1_macro"],
            "Weighted F1": result["f1_weighted"],
            "Train Time (s)": result["train_time"],
            "Inference Time (s)": result["inference_time"],
            "Fantasy Recall": result["fantasy_recall"]
        })

    df_results = pd.DataFrame(results)

    print("\nRESUMO FINAL:")
    print(df_results.round(4).to_string(index=False))
    for model_name, result in predictions_dict.items():
        print("\n" + "=" * 70)
        print(model_name)
        print("=" * 70)

        report_df = pd.DataFrame(result["classification_report"]).transpose()
        report_df = report_df.round(4)
        filename = re.sub(r"[^\w]+", "_", model_name).strip("_")
        report_df.to_csv(f"{save_prefix}_{filename}_classification_report.csv")
        print(report_df)

    df_results.to_csv(f"{save_prefix}_benchmark_results.csv", index=False)

    # -----------------------------------------------------
    # Gráfico de métricas
    # -----------------------------------------------------

    metrics_plot = df_results[
        [
            "Model",
            "Accuracy",
            "Precision Macro",
            "Recall Macro",
            "Macro F1",
            "Weighted F1",
            "Fantasy Recall"
        ]
    ]

    df_melted = metrics_plot.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(15, 6))

    sns.barplot(
        data=df_melted,
        x="Model",
        y="Score",
        hue="Metric",
        palette="viridis"
    )

    plt.ylim(0, 1.05)

    plt.title(
        "Benchmark Tipográfico - Comparação dos Modelos",
        fontsize=14
    )

    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    plt.savefig(f"{save_prefix}_metrics.pdf", format="pdf", bbox_inches="tight")

    plt.close()

    # -----------------------------------------------------
    # Matrizes de confusão
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10)
    )

    axes = axes.flatten()

    for i, (model_name, result) in enumerate(predictions_dict.items()):

        y_pred = result["predictions"]

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            ax=axes[i]
        )

        axes[i].set_title(model_name)

        axes[i].set_xlabel("Predição")

        axes[i].set_ylabel("Classe verdadeira")

    plt.tight_layout()

    plt.savefig(f"{save_prefix}_confusion.pdf", format="pdf", bbox_inches="tight")

    plt.close()

    print("\nArquivos gerados:")
    print(f"  • {save_prefix}_benchmark_results.csv")
    print(f"  • {save_prefix}_metrics.pdf")
    print(f"  • {save_prefix}_confusion.pdf")

    return df_results

def train_dl_model(config_name, train_loader, val_loader, num_classes=4, epochs=10, patience=5):
    print(f"\nTreinando {config_name} por {epochs} épocas...")
    
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=2)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    best_acc = -float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0

    epochs_without_improvement = 0
    
    epoch_iterator = tqdm(range(epochs), desc=f"Épocas ({config_name})", unit="época")
    train_start = time.perf_counter()
    for epoch in epoch_iterator:
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)

        # -----------------------------
        # Validação
        # -----------------------------
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_acc)

        if val_acc >= best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_iterator.set_postfix({
        "Train Loss": f"{epoch_loss:.4f}",
        "Val Loss": f"{val_loss:.4f}",
        "Val Acc": f"{val_acc:.4f}",
        "Best": f"{best_acc:.4f}",
        "LR": f"{optimizer.param_groups[0]['lr']:.1e}"
        })

        if epochs_without_improvement >= patience:
            print(f"\nEarly Stopping na época {epoch+1}")
            break
        
    model.load_state_dict(best_weights)
    print(f"\nMelhor acurácia de validação: {best_acc:.4f}")
    print(f"Melhor modelo salvo na época {best_epoch}")

    train_time = time.perf_counter() - train_start
    print(f"Tempo de treinamento: {train_time:.2f} s")
    return model, train_time


def run_final_test(df):
    print("\n" + "=" * 70)
    print("TREINAMENTO FINAL (HOLD-OUT SPLIT SIMPLES)")
    print("=" * 70)

    class_names = sorted(df["label"].unique())

    # 1. Separar o Conjunto de Teste Inicial (ex: 15% dos dados)
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df["label_encoded"],
        random_state=SEED
    )

    # 2. Dividir o restante entre Treino e Validação (85% restantes -> dividido novamente)
    # Por exemplo, test_size=0.1764 de 85% resulta em ~15% de validação no total
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.1764, # Ajuste para ter proporções Treino(70) / Val(15) / Test(15)
        stratify=train_val_df["label_encoded"],
        random_state=SEED
    )
    
    print(f"Total de imagens - Treino: {len(train_df)} | Validação: {len(val_df)} | Teste: {len(test_df)}")

    # ----------------------------------------------------
    # DataLoaders
    # ----------------------------------------------------
    g = torch.Generator()
    g.manual_seed(SEED)

    loader_train = DataLoader(
        TypographyDataset(train_df, transform=base_transform),
        batch_size=32,
        shuffle=True,
        generator=g
    )

    loader_val = DataLoader(
        TypographyDataset(val_df, transform=base_transform),
        batch_size=32,
        shuffle=False
    )

    loader_test = DataLoader(
        TypographyDataset(test_df, transform=base_transform),
        batch_size=32,
        shuffle=False
    )

    # ----------------------------------------------------
    # Treinamento
    # ----------------------------------------------------
    model, train_time = train_dl_model(
        "ResNet-18 (Final)",
        loader_train,
        loader_val,
        epochs=30
    )

    # ----------------------------------------------------
    # Avaliação
    # ----------------------------------------------------
    y_true = test_df["label_encoded"].values

    result = evaluate_model(
        model=model,
        X_test=None,
        class_names=class_names,
        y_test=y_true,
        model_type="pytorch",
        test_loader=loader_test,
        train_time=train_time
    )

    evaluate_all_models(
        y_true=y_true,
        predictions_dict={
            "ResNet-18 (Final)": result
        },
        class_names=class_names,
        save_prefix="final_test"
    )
    print("\n" + "=" * 70)
    print("TREINAMENTO FINAL")
    print("=" * 70)

    # ----------------------------------------------------
    # Train = folds 0..4
    # Test = fold -1
    # ----------------------------------------------------
    class_names = sorted(df["label"].unique())


    train_df = df[df["fold"] != -1].copy()
    test_df  = df[df["fold"] == -1].copy()


    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df["label_encoded"],
        random_state=SEED
    )
    # ----------------------------------------------------
    # DataLoaders
    # ----------------------------------------------------

    loader_train = DataLoader(
        TypographyDataset(train_df, transform=base_transform),
        batch_size=32,
        shuffle=True
    )

    loader_val = DataLoader(
        TypographyDataset(val_df, transform=base_transform),
        batch_size=32,
        shuffle=False
    )

    loader_test = DataLoader(
        TypographyDataset(test_df, transform=base_transform),
        batch_size=32,
        shuffle=False
    )

    # ----------------------------------------------------
    # Treinamento
    # ----------------------------------------------------

    model, train_time = train_dl_model(
        "ResNet-18 (Final)",
        loader_train,
        loader_val,
        epochs=30
    )

    # ----------------------------------------------------
    # Avaliação
    # ----------------------------------------------------

    y_true = test_df["label_encoded"].values

    result = evaluate_model(
        model=model,
        X_test=None,
        class_names=class_names,
        y_test=y_true,
        model_type="pytorch",
        test_loader=loader_test,
        train_time=train_time
    )

    evaluate_all_models(
        y_true=y_true,
        predictions_dict={
            "ResNet-18 (Final)": result
        },
        class_names=class_names,
        save_prefix="final_test"
    )

if __name__ == "__main__":
    SPLITS_CSV_PATH = "../../data/k-fold_split/typography_dataset_splits.csv"
    df = pd.read_csv(SPLITS_CSV_PATH)

    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])

    run_final_test(df)