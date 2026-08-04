import time
import cv2
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
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from skimage.feature import hog
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

# =========================================================
# 1. CLASSES E TRANSFORMS (DATASETS)
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

train_transform_aug = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15), 
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)), 
    transforms.ColorJitter(brightness=0.3, contrast=0.3), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================================================
# 2. EXTRAÇÃO PARA MACHINE LEARNING CLÁSSICO
# =========================================================
def extract_classical_features(dataframe, target_size=(64, 64), method="hog"):
    features = []
    labels = []
    
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Extraindo características ({method})"):
        img_path = row['image_path']
        label = row['label_encoded'] # Usando label_encoded diretamente
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        img_resized = cv2.resize(img, target_size)
        if method.lower() == 'hog':
            feature_vector = hog(
            img_resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
            )

        elif method.lower() == "flatten":
            feature_vector = img_resized.flatten()

        else:
            raise ValueError(
                f"Método '{method}' não suportado. "
                "Use 'hog' ou 'flatten'."
            )

        features.append(feature_vector)
        labels.append(label)
        
    return np.array(features), np.array(labels)

# =========================================================
# 3. TREINAMENTO DE DEEP LEARNING (AS CONFIGURAÇÕES EXATAS)
# =========================================================
def train_dl_model(config_name, train_loader, val_loader, num_classes=4, epochs=10, patience=5):
    print(f"\nTreinando {config_name} por {epochs} épocas...")
    
    if "ResNet" in config_name:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    elif "EfficientNet" in config_name:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
        # Conforme seu código: lr 0.0001 com weight_decay
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

# =========================================================
# 4. AVALIAÇÃO E GERAÇÃO DOS GRÁFICOS (PARA O LATEX)
# =========================================================
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

def run_cross_validation(df, n_folds=5):

    all_results = []

    for val_fold in range(n_folds):

        fold_results = run_fold(
            df,
            val_fold=val_fold
        )

        fold_results["Fold"] = val_fold

        all_results.append(fold_results)

    results = pd.concat(
        all_results,
        ignore_index=True
    )

    results.to_csv(
        "cross_validation_results.csv",
        index=False
    )

    summary = (
    results
    .groupby("Model")
    .agg(["mean", "std"])
    .round(4)
    )

    summary.columns = [
    f"{metric}_{stat}"
    for metric, stat in summary.columns
    ]

    print("\n===== MÉDIAS DOS 5 FOLDS =====")
    print(summary)

    summary.to_csv("cross_validation_summary.csv")

    return results, summary

def run_fold(df, val_fold, test_fold=-1):

    print(f"\n{'='*70}")
    print(f"VALIDATION FOLD {val_fold}")
    print(f"{'='*70}")

    train_df = df[
        (df["fold"] != val_fold) &
        (df["fold"] != test_fold)
    ].copy()

    val_df = df[df["fold"] == val_fold].copy()
    test_df = df[df["fold"] == test_fold].copy()

    class_names = sorted(df["label"].unique())

    # --- PREPARANDO LOADERS DE IMAGENS ---
            # Para testes e modelos sem augmentation
    g = torch.Generator()
    g.manual_seed(SEED)
    loader_base_train = DataLoader(
        TypographyDataset(train_df, transform=base_transform),
        batch_size=32,
        shuffle=True,
        generator=g
    ) 
    
    loader_aug_train = DataLoader(
        TypographyDataset(train_df, transform=train_transform_aug),
        batch_size=32,
        shuffle=True,
        generator=g
    ) 
    # Validação (SEM augmentation)
    loader_val = DataLoader(
        TypographyDataset(val_df, transform=base_transform),
        batch_size=32,
        shuffle=False
    ) 
    # Teste (SEM augmentation)
    loader_test = DataLoader(
        TypographyDataset(test_df, transform=base_transform),
        batch_size=32,
        shuffle=False
    ) 
    # Dicionário que guardará todas as previsões
    model_preds = {}
    # --- 1 & 2. MACHINE LEARNING CLÁSSICO ---
    print("\n--- PREPARANDO MACHINE LEARNING CLÁSSICO ---")
    X_train, y_train = extract_classical_features(train_df, target_size=(64, 64), method='hog')
    X_test, y_true_test = extract_classical_features(test_df, target_size=(64, 64), method='hog')

    print("Treinando Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=0)
    start = time.perf_counter()
    rf.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    model_preds['1. Random Forest'] = evaluate_model(
        rf,
        X_test,
        class_names,
        y_true_test,
        model_type="sklearn",
        train_time=train_time
    )
    print("Treinando XGBoost...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.001, max_depth=6, random_state=42, n_jobs=-1)
    start = time.perf_counter()
    xgb.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    model_preds['2. XGBoost'] = evaluate_model(
        xgb,
        X_test,
        class_names,
        y_true_test,
        model_type="sklearn",
        train_time=train_time
    )
    # --- 3. RESNET-18 (SEM AUG, 10 ÉPOCAS) ---
    print("\n--- INICIANDO DEEP LEARNING ---")
    res_base, train_time = train_dl_model('3. ResNet-18 (Base)', loader_base_train, loader_val, epochs=30)
    model_preds["3. ResNet-18 (Base)"] = evaluate_model(
        res_base,
        None,
        class_names,
        y_true_test,
        model_type="pytorch",
        test_loader=loader_test,
        train_time=train_time
    ) 
    # --- 4. RESNET-18 (COM AUG, 40 ÉPOCAS) ---
    res_aug, train_time = train_dl_model('4. ResNet-18 (Aug)', loader_aug_train, loader_val, epochs=40)
    model_preds["4. ResNet-18 (Aug)"] = evaluate_model(
        res_aug,
        None,
        class_names,
        y_true_test,
        model_type="pytorch",
        test_loader=loader_test,
        train_time=train_time
    )
    # --- 5. EFFICIENTNET-B0 (SEM AUG, 10 ÉPOCAS) ---
    eff_base, train_time = train_dl_model('5. EfficientNet (Base)', loader_base_train, loader_val, epochs=30)
    model_preds["5. EfficientNet (Base)"] = evaluate_model(
        eff_base,
        None,
        class_names,
        y_true_test,
        model_type="pytorch",
        test_loader=loader_test,
        train_time=train_time
    )
    # --- 6. EFFICIENTNET-B0 (COM AUG, 40 ÉPOCAS) ---
    # Assumi 40 épocas para igualar com a ResNet com augmentation
    eff_aug, train_time = train_dl_model('6. EfficientNet (Aug)', loader_aug_train, loader_val, epochs=45)
    model_preds["6. EfficientNet (Aug)"] = evaluate_model(
        eff_aug,
        None,
        class_names,
        y_true_test,
        model_type="pytorch",
        test_loader=loader_test,
        train_time=train_time
    )
    # --- CONFRONTO FINAL ---
    return evaluate_all_models(y_true_test, model_preds, class_names, save_prefix=f"fold_{val_fold}")


if __name__ == "__main__":
    SPLITS_CSV_PATH = "../../data/k-fold_split/typography_dataset_splits.csv"
    df = pd.read_csv(SPLITS_CSV_PATH)

    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])

    run_cross_validation(df)

        
