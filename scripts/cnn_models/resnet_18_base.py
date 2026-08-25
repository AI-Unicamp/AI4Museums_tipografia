import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm

ROOT_DIRECTORY = '../../data'
SPLITS_CSV_PATH = ROOT_DIRECTORY + '/k-fold_split/typography_dataset_splits_local.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TypographyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, 'image_path']
        label = self.dataframe.loc[idx, 'label_encoded']
        
        # Carrega a imagem e garante que seja RGB (3 canais) para a ResNet
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Erro ao carregar a imagem {img_path}: {e}")
            # Retorna uma imagem preta em caso de erro para não quebrar o loop
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# ==========================================
# 2. DEFINIÇÃO DAS TRANSFORMAÇÕES
# ==========================================
# SEM Data Augmentation - Apenas redimensionamento e normalização padrão do ImageNet
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==========================================
# 3. TREINAMENTO DA RESNET-18
# ==========================================
def train_resnet(train_loader, val_loader, num_classes=4, epochs=30, patience=5):

    print(f"\nTreinando ResNet-18 por até {epochs} épocas...")

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = -float("inf")
    best_weights = None
    best_epoch = 0
    epochs_without_improvement = 0

    start_time = time.time()

    # Adicionando o tqdm aqui
    epoch_iterator = tqdm(range(epochs), desc="Progresso das Épocas", unit="época")

    for epoch in epoch_iterator:
        # =========================
        # TREINAMENTO
        # =========================
        model.train()

        running_loss = 0.0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # =========================
        # VALIDAÇÃO
        # =========================
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total

        # Atualizando as métricas na barra do tqdm em vez do print
        epoch_iterator.set_postfix({
            "Train Loss": f"{train_loss:.4f}",
            "Val Acc": f"{val_accuracy:.4f}"
        })

        # =========================
        # MELHOR MODELO
        # =========================
        if val_accuracy > best_acc:

            best_acc = val_accuracy

            best_weights = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }

            best_epoch = epoch + 1
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping na época {epoch + 1}.")
            break

    # Restaurar melhor modelo
    model.load_state_dict(best_weights)
    model = model.to(device)

    MODEL_PATH = "resnet18_final.pth"

    torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nModelo salvo em: {MODEL_PATH}")

    training_time = time.time() - start_time

    print(f"Melhor época: {best_epoch}")
    print(f"Melhor validação: {best_acc:.4f}")
    print(f"Tempo de treinamento: {training_time:.2f}s")

    return model, training_time


def run_final_test(csv_path, batch_size=32, epochs=30):

    print("\n" + "=" * 60)
    print("TREINAMENTO FINAL - RESNET-18")
    print("=" * 60)

    # ==========================================
    # CARREGAR DATASET
    # ==========================================

    df = pd.read_csv(csv_path)

    # ==========================================
    # LABEL ENCODER
    # ==========================================

    classes = sorted([
        'grotesco',
        'serifado',
        'escritural',
        'fantasia'
    ])

    label_encoder = LabelEncoder()
    label_encoder.fit(classes)

    df['label_encoded'] = label_encoder.transform(df['label'])

    # ==========================================
    # SEPARAÇÃO TREINO / TESTE FINAL
    # ==========================================

    train_df = df[df['fold'] != -1].copy()

    test_df = df[df['fold'] == -1].copy()

    if test_df.empty:
        raise ValueError("Nenhuma imagem encontrada no fold -1.")

    if train_df.empty:
        raise ValueError("Nenhuma imagem encontrada para treinamento.")

    expected_classes = set(classes)
    if set(test_df['label']) != expected_classes:
        print("\nAVISO: o conjunto de teste não contém todas as classes.")

    print(f"\nImagens para treinamento: {len(train_df)}")
    print(f"Imagens para teste final: {len(test_df)}")

    print("\nDistribuição do treinamento:")
    print(train_df['label'].value_counts())

    print("\nDistribuição do teste:")
    print(test_df['label'].value_counts())

    # ==========================================
    # SEPARAÇÃO TREINO / VALIDAÇÃO
    # ==========================================

    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df['label_encoded'],
        random_state=42
    )

    print(f"\nApós separação:")
    print(f"Treinamento: {len(train_df)}")
    print(f"Validação:   {len(val_df)}")
    print(f"Teste:       {len(test_df)}")

    # ==========================================
    # DATASETS
    # ==========================================

    train_dataset = TypographyDataset(
        train_df,
        transform=base_transform
    )

    val_dataset = TypographyDataset(
        val_df,
        transform=base_transform
    )

    test_dataset = TypographyDataset(
        test_df,
        transform=base_transform
    )

    # ==========================================
    # DATALOADERS
    # ==========================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # ==========================================
    # TREINAMENTO
    # ==========================================

    model, training_time = train_resnet(
        train_loader,
        val_loader,
        num_classes=4,
        epochs=epochs,
        patience=5
    )

    # ==========================================
    # TESTE FINAL
    # ==========================================

    print("\n" + "=" * 60)
    print("TESTE FINAL")
    print("=" * 60)

    model.eval()

    y_true = []
    y_pred = []

    start_time = time.time()

    with torch.no_grad():

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())

    inference_time = time.time() - start_time

    # ==========================================
    # RESULTADOS
    # ==========================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print(f"\nAccuracy final: {accuracy:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3],
            target_names=classes,
            zero_division=0
        )
    )

    print(f"Tempo de treinamento: {training_time:.2f}s")

    print(f"Tempo de inferência: {inference_time:.2f}s")

    return model


if __name__ == "__main__":
    if os.path.exists(SPLITS_CSV_PATH):
        run_final_test(SPLITS_CSV_PATH,batch_size=32,epochs=30)
    else:
        print("Erro: CSV não encontrado.")