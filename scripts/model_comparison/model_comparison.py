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
import torchvision.transforms.functional as TF
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
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

class RandomAffineBorderFill:
    """
    Transformação afim leve (rotação, translação, escala e cisalhamento pequenos)
    que preenche as bordas reveladas com a COR MEDIANA DA BORDA da própria imagem.

    O RandomAffine padrão preenche com preto (fill=0), o que cria cantos pretos
    inexistentes nas letras reais (recortadas de papel cinza/amarelado) e gera
    um viés artificial entre treino e inferência.
    """
    def __init__(self, degrees=5, translate=(0.05, 0.05), scale=(0.92, 1.08), shear=3, p=0.7):
        self.degrees = (-degrees, degrees)
        self.translate = translate
        self.scale = scale
        self.shear = (-shear, shear)
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        angle, translations, scale, shear = transforms.RandomAffine.get_params(
            self.degrees, self.translate, self.scale, self.shear, list(img.size)
        )

        arr = np.asarray(img)
        border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0)
        fill = tuple(int(v) for v in np.median(border, axis=0))

        return TF.affine(
            img,
            angle=angle,
            translate=list(translations),
            scale=scale,
            shear=list(shear),
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=fill,
        )


class AddGaussianNoise:
    """Ruído gaussiano aditivo sobre o tensor em [0, 1] (aplicar ANTES do Normalize)."""
    def __init__(self, std=0.03, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
        return (tensor + torch.randn_like(tensor) * self.std).clamp(0.0, 1.0)


# Augmentation que preserva a geometria tipográfica:
#  - SEM flips (horizontal/vertical): espelhar 'b'/'d', 'p'/'q', itálicos etc. gera ruído de rótulo.
#  - SEM RandomResizedCrop: ele altera a proporção (aspect ratio) do recorte e pode cortar
#    serifas — justamente o tipo de distorção que já vimos causar desvio treino/inferência.
#  - Resize((224, 224)) PRIMEIRO, idêntico ao base_transform, para o modelo ver a mesma
#    geometria base e as variações serem apenas pequenas perturbações em torno dela.
train_transform_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    RandomAffineBorderFill(degrees=5, translate=(0.05, 0.05), scale=(0.92, 1.08), shear=3, p=0.7),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    AddGaussianNoise(std=0.03, p=0.5),
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
def compute_class_weights(train_df, num_classes, power=1.0):
    """
    Pesos inversamente proporcionais à frequência de cada classe NO TREINO DO FOLD:
        w_c = (N / (K * n_c)) ** power
    (power=1.0 equivale ao 'balanced' do scikit-learn; power=0.5 dá uma ponderação
    mais suave, útil se a loss ficar instável com a classe rara.)
    """
    counts = np.bincount(train_df["label_encoded"].values, minlength=num_classes).astype(float)
    weights = (counts.sum() / (num_classes * np.maximum(counts, 1.0))) ** power
    return torch.tensor(weights, dtype=torch.float32)


def train_dl_model(config_name, train_loader, val_loader, num_classes=4, epochs=10, patience=5,
                   class_weights=None, selection_metric="macro_f1"):
    """
    class_weights: tensor (num_classes,) com os pesos da entropia cruzada (ou None).
    selection_metric: métrica de validação usada para escolher o melhor checkpoint,
        reduzir o LR e disparar o early stopping. 'macro_f1' (padrão) ou 'accuracy'.
        Com classes desbalanceadas, escolher por acurácia favorece a classe majoritária
        e anula o efeito da loss ponderada.
    """
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

    if class_weights is not None:
        print(f"Pesos de classe na loss: {[round(w, 3) for w in class_weights.tolist()]}")
        class_weights = class_weights.to(device)

    # Loss de treino ponderada; loss de validação SEM pesos (só para monitoramento/comparabilidade).
    # Obs.: com pesos, o "Train Loss" exibido é uma média aproximada (o PyTorch normaliza pela soma
    # dos pesos de cada lote) — serve apenas para acompanhar a curva.
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    eval_criterion = nn.CrossEntropyLoss()

    best_score = -float("inf")
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
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, eval_criterion, num_classes)
        val_score = val_f1 if selection_metric == "macro_f1" else val_acc
        scheduler.step(val_score)

        if val_score >= best_score:
            best_score = val_score
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
        "Best": f"{best_score:.4f}",
        "LR": f"{optimizer.param_groups[0]['lr']:.1e}"
        })

        if epochs_without_improvement >= patience:
            print(f"\nEarly Stopping na época {epoch+1}")
            break
        
    model.load_state_dict(best_weights)
    print(f"\nMelhor {selection_metric} de validação: {best_score:.4f}")
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


def evaluate_model(model, X_test, class_names, y_test, model_type="sklearn", test_loader=None, train_time=None,
                   feature_time=0.0):
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

    # Para os modelos clássicos, `feature_time` é o tempo de leitura + resize + HOG das imagens de teste,
    # que acontece fora do predict(). Somamos aqui para ficar comparável ao tempo dos modelos de DL,
    # cujo predict_dl_model já inclui leitura do disco e transformações via DataLoader.
    inference_time = (time.perf_counter() - start) + feature_time

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
    

def evaluate(model, loader, criterion, num_classes=4):
    """
    Avalia o modelo em um DataLoader.
    Retorna:
        loss médio
        accuracy
        macro-F1 (todas as classes contam igual; classe sem acerto entra com F1 = 0)
    """

    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    loss = running_loss / len(all_labels)
    accuracy = float((all_preds == all_labels).mean())
    macro_f1 = f1_score(
        all_labels, all_preds,
        labels=np.arange(num_classes), average="macro", zero_division=0
    )

    return loss, accuracy, macro_f1

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

# =========================================================
# 5. VERIFICAÇÃO DOS FOLDS E RESULTADOS AGREGADOS
# =========================================================
def check_fold_coverage(df, class_names):
    """Falha cedo se algum fold estiver sem exemplos de alguma classe."""
    table = pd.crosstab(df["fold"], df["label"]).reindex(columns=class_names, fill_value=0)
    print("\nImagens por classe em cada fold:")
    print(table.to_string())
    if (table.values == 0).any():
        raise ValueError(
            "Há folds sem nenhuma imagem de alguma classe (ver tabela acima). "
            "Regenere o CSV com dataset_split_kfold.py antes de rodar a validação cruzada."
        )


def _draw_confusion(ax, cm, cm_norm, class_names, title):
    """Heatmap normalizado por linha (recall por classe) anotado com % e contagem."""
    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_norm[i, j] * 100:.1f}%\n(n={cm[i, j]})"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Predição")
    ax.set_ylabel("Classe verdadeira")


def report_aggregated_results(oof_true, oof_preds, class_names, results_dir=Path("../../results")):
    """
    Junta as predições dos 5 folds de teste (cada imagem é testada exatamente uma vez,
    então o conjunto agregado cobre o dataset inteiro, sem sobreposição) e gera:
      - matriz de confusão agregada NORMALIZADA por linha (PDF em grade + um PDF por modelo);
      - contagens brutas da matriz (CSV);
      - classification report agregado por modelo (CSV) e resumo comparativo (CSV).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.concatenate(oof_true)
    labels = np.arange(len(class_names))
    summary_rows = []

    n_models = len(oof_preds)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (model_name, pred_list) in zip(axes, oof_preds.items()):
        y_pred = np.concatenate(pred_list)
        assert len(y_pred) == len(y_true), f"Tamanhos diferentes em '{model_name}'"

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

        safe_name = re.sub(r"[^\w]+", "_", model_name).strip("_")

        # Contagens brutas (para tabelas no LaTeX)
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
            results_dir / f"aggregated_{safe_name}_confusion_counts.csv"
        )

        # Classification report agregado
        report = classification_report(
            y_true, y_pred, labels=labels, target_names=class_names,
            output_dict=True, zero_division=0
        )
        pd.DataFrame(report).transpose().round(4).to_csv(
            results_dir / f"aggregated_{safe_name}_classification_report.csv"
        )
        summary_rows.append({
            "Model": model_name,
            "Accuracy": report["accuracy"],
            "Macro F1": report["macro avg"]["f1-score"],
            "Weighted F1": report["weighted avg"]["f1-score"],
            "Fantasy Recall": report["fantasia"]["recall"],
            "N": len(y_true),
        })

        # Painel na grade
        _draw_confusion(ax, cm, cm_norm, class_names, model_name)

        # Figura individual (para o relatório)
        fig_single, ax_single = plt.subplots(figsize=(7, 5.5))
        _draw_confusion(ax_single, cm, cm_norm, class_names, f"{model_name} — agregado (5 folds)")
        fig_single.tight_layout()
        fig_single.savefig(results_dir / f"aggregated_{safe_name}_confusion.pdf",
                           format="pdf", bbox_inches="tight")
        plt.close(fig_single)

    for ax in axes[n_models:]:
        ax.axis("off")

    fig.suptitle("Matrizes de confusão agregadas (5 folds de teste, normalizadas por classe verdadeira)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(results_dir / "aggregated_confusion_all_models.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_dir / "aggregated_summary.csv", index=False)

    print("\n===== RESULTADOS AGREGADOS (predições dos 5 folds de teste juntas) =====")
    print(summary.round(4).to_string(index=False))
    print(f"\nArquivos 'aggregated_*' salvos em {results_dir.resolve()}")
    return summary


def run_cross_validation(df, n_folds=5):

    class_names = sorted(df["label"].unique())
    check_fold_coverage(df, class_names)

    all_results = []
    oof_true = []      # y_true de cada fold de teste, na ordem
    oof_preds = {}     # modelo -> lista de predições de cada fold de teste

    for test_fold in range(n_folds):

        val_fold = (test_fold + 1) % n_folds
        
        fold_results, y_true_fold, fold_preds = run_fold(
            df,
            test_fold=test_fold,
            val_fold=val_fold
        )

        fold_results["Test_Fold"] = test_fold
        fold_results["Val_Fold"] = val_fold
        all_results.append(fold_results)

        oof_true.append(np.asarray(y_true_fold))
        for model_name, result in fold_preds.items():
            oof_preds.setdefault(model_name, []).append(np.asarray(result["predictions"]))

    results = pd.concat(all_results, ignore_index=True)

    results.to_csv("cross_validation_results.csv", index=False
    )

    summary = (
    results
    .groupby("Model")
    .agg(["mean", "std"])
    .round(4)
    )

    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]

    print("\n===== MÉDIAS DOS 5 FOLDS =====")
    print(summary)

    summary.to_csv("cross_validation_summary.csv")

    # Matriz de confusão e métricas agregadas sobre todo o dataset (fora-da-amostra)
    report_aggregated_results(oof_true, oof_preds, class_names)

    return results, summary

def run_fold(df, val_fold, test_fold=-1):

    print(f"\n{'='*70}")
    print(f"ITERATION -> TEST FOLD: {test_fold} | VAL FOLD: {val_fold}")
    print(f"{'='*70}")

    train_df = df[
        (df["fold"] != val_fold) &
        (df["fold"] != test_fold)
    ].copy()

    val_df = df[df["fold"] == val_fold].copy()
    test_df = df[df["fold"] == test_fold].copy()

    class_names = sorted(df["label"].unique())
    num_classes = len(class_names)

    # Pesos de classe calculados só com o TREINO deste fold (sem vazar val/teste)
    class_weights = compute_class_weights(train_df, num_classes)

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
    # Tempo da extração (leitura + resize + HOG) é medido e somado aos tempos dos modelos clássicos.
    # A extração é feita uma vez e compartilhada por RF e XGBoost; cada um paga o custo inteiro,
    # já que nenhum funcionaria sem ela.
    t0 = time.perf_counter()
    X_train, y_train = extract_classical_features(train_df, target_size=(64, 64), method='hog')
    train_feat_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_test, y_true_test = extract_classical_features(test_df, target_size=(64, 64), method='hog')
    test_feat_time = time.perf_counter() - t0
    print(f"Extração de HOG: treino {train_feat_time:.2f} s ({len(train_df)} imgs) | "
          f"teste {test_feat_time:.2f} s ({len(test_df)} imgs)")
    # Garante alinhamento com as predições dos loaders (imagens ilegíveis quebrariam isso)
    assert len(y_true_test) == len(test_df), (
        "Alguma imagem de teste não pôde ser lida por cv2.imread; "
        "y_true e as predições dos modelos de DL ficariam desalinhados."
    )

    print("Treinando Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=0,
                                class_weight="balanced")
    start = time.perf_counter()
    rf.fit(X_train, y_train)
    fit_time = time.perf_counter() - start
    train_time = train_feat_time + fit_time
    print(f"Random Forest: fit {fit_time:.2f} s + extração HOG {train_feat_time:.2f} s = {train_time:.2f} s")
    model_preds['1. Random Forest'] = evaluate_model(
        rf,
        X_test,
        class_names,
        y_true_test,
        model_type="sklearn",
        train_time=train_time,
        feature_time=test_feat_time
    )
    print("Treinando XGBoost...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    start = time.perf_counter()
    xgb.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    fit_time = time.perf_counter() - start
    train_time = train_feat_time + fit_time
    print(f"XGBoost: fit {fit_time:.2f} s + extração HOG {train_feat_time:.2f} s = {train_time:.2f} s")
    model_preds['2. XGBoost'] = evaluate_model(
        xgb,
        X_test,
        class_names,
        y_true_test,
        model_type="sklearn",
        train_time=train_time,
        feature_time=test_feat_time
    )
    # --- 3. RESNET-18 (SEM AUG, 10 ÉPOCAS) ---
    print("\n--- INICIANDO DEEP LEARNING ---")
    res_base, train_time = train_dl_model('3. ResNet-18 (Base)', loader_base_train, loader_val, epochs=30,
                                         num_classes=num_classes, class_weights=class_weights)
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
    res_aug, train_time = train_dl_model('4. ResNet-18 (Aug)', loader_aug_train, loader_val, epochs=40,
                                         num_classes=num_classes, class_weights=class_weights)
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
    eff_base, train_time = train_dl_model('5. EfficientNet (Base)', loader_base_train, loader_val, epochs=30,
                                         num_classes=num_classes, class_weights=class_weights)
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
    eff_aug, train_time = train_dl_model('6. EfficientNet (Aug)', loader_aug_train, loader_val, epochs=45,
                                         num_classes=num_classes, class_weights=class_weights)
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
    df_results = evaluate_all_models(y_true_test, model_preds, class_names, save_prefix=f"test_fold{test_fold}")
    return df_results, y_true_test, model_preds


if __name__ == "__main__":
    SPLITS_CSV_PATH = "../../data/k-fold_split/typography_dataset_splits.csv"
    df = pd.read_csv(SPLITS_CSV_PATH)

    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])

    run_cross_validation(df)

        
