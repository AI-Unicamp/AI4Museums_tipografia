import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm

ROOT_DIRECTORY = '../../data'
SPLITS_CSV_PATH = ROOT_DIRECTORY + '/k-fold_split/typography_dataset_splits_local.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

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


def load_fold_from_csv(csv_path, target_fold_idx, label_encoder):
    df = pd.read_csv(csv_path)
    
    # Transforma os rótulos de texto em números e adiciona ao DataFrame
    df['label_encoded'] = label_encoder.transform(df['label'])
    
    fold_df = df[df['fold'] == target_fold_idx]
    train_df = fold_df[fold_df['split_type'] == 'train']
    val_df = fold_df[fold_df['split_type'] == 'val']
    
    return train_df, val_df

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
# 3. LOOP DE TREINAMENTO DO FOLD
# ==========================================
def train_efficientnet_fold(train_loader, val_loader, num_classes=4, epochs=10):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    epoch_iterator = tqdm(range(epochs), desc="Treinando Épocas", unit="época")
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
        epoch_iterator.set_postfix({"Loss": f"{epoch_loss:.4f}"})
            
    # Fase de Validação final do fold
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_labels, all_preds

# ==========================================
# 4. EXECUÇÃO DA VALIDAÇÃO CRUZADA
# ==========================================
def run_cv_efficientnet(csv_path, num_folds=5, batch_size=32, epochs=5):
    print("Iniciando Validação Cruzada com EfficientNet-B0 (Sem Augmentation)...")
    
    classes = sorted(['grotesco', 'serifado', 'escritural', 'fantasia'])
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    
    fold_metrics = []
    
    for fold_idx in range(num_folds):
        print(f"\n{'='*40}")
        print(f"Processando Fold {fold_idx + 1}/{num_folds} (EfficientNet-B0)")
        print(f"{'='*40}")
        
        train_df, val_df = load_fold_from_csv(csv_path, fold_idx, label_encoder)
        
        train_dataset = TypographyDataset(train_df, transform=base_transform)
        val_dataset = TypographyDataset(val_df, transform=base_transform)
        
        # DataLoaders gerenciam o envio de lotes de imagens para o modelo
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        start_time = time.time()
        
        # Treinando com poucas épocas (5) para o teste inicial, pois a rede já é pré-treinada
        y_true, y_pred = train_efficientnet_fold(train_loader, val_loader, num_classes=4, epochs=epochs)
        
        training_time = time.time() - start_time
        fold_accuracy = accuracy_score(y_true, y_pred)
        
        print(f"Fold {fold_idx + 1} Acurácia: {fold_accuracy:.4f} (Tempo: {training_time:.2f}s)")
        print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3], target_names=classes, zero_division=0))
        
        fold_metrics.append(fold_accuracy)
        
    avg_accuracy = np.mean(fold_metrics)
    std_accuracy = np.std(fold_metrics)
    
    print(f"\n{'='*40}")
    print("RESULTADO FINAL CROSS-VALIDATION - EFFICIENT NET B0")
    print(f"Acurácia Média: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"{'='*40}\n")

    if __name__ == "__main__":
        if os.path.exists(SPLITS_CSV_PATH):
            run_cv_efficientnet(SPLITS_CSV_PATH, num_folds=5, batch_size=32, epochs=25)
        else:
            print("Erro: CSV não encontrado.")