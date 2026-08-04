import os
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
from tqdm import tqdm

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
def extract_classical_features(dataframe, target_size=(64, 64)):
    features = []
    labels = []
    
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Extraindo imagens ML Clássico"):
        img_path = row['image_path']
        label = row['label_encoded'] # Usando label_encoded diretamente
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        img_resized = cv2.resize(img, target_size)
        flattened_features = img_resized.flatten()
        
        features.append(flattened_features)
        labels.append(label)
        
    return np.array(features), np.array(labels)

# =========================================================
# 3. TREINAMENTO DE DEEP LEARNING (AS CONFIGURAÇÕES EXATAS)
# =========================================================
def train_dl_model(config_name, train_loader, num_classes=4, epochs=10):
    print(f"\nTreinando {config_name} por {epochs} épocas...")
    
    if "ResNet" in config_name:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        # LR baseado na sua especificação
        lr = 0.0001 if "Augmentation" in config_name else 0.001
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    elif "EfficientNet" in config_name:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
        # Conforme seu código: lr 0.0001 com weight_decay
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    epoch_iterator = tqdm(range(epochs), desc=f"Épocas ({config_name})", unit="época")
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
        
    return model

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

# =========================================================
# 4. AVALIAÇÃO E GERAÇÃO DOS GRÁFICOS (PARA O LATEX)
# =========================================================
def compare_models(y_true, predictions_dict, class_names):
    print("\n" + "="*50)
    print("GERANDO RESULTADOS COMPARTIVOS E PDFs")
    print("="*50)
    
    results = []
    idx_fantasia = class_names.index('fantasia') 
    
    for model_name, y_pred in predictions_dict.items():
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
        recall_fantasia = recalls[idx_fantasia]
        
        results.append({
            'Model': model_name,
            'Accuracy': acc,
            'Macro F1': f1_macro,
            'Fantasy Recall': recall_fantasia
        })
        
    df_results = pd.DataFrame(results)
    print("\nRESUMO FINAL:")
    print(df_results.to_string(index=False))
    
    # 1. Gráfico de Barras
    df_melted = df_results.melt(id_vars='Model', var_name='Metric', value_name='Score')
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', palette='viridis')
    plt.title('Benchmark Tipográfico: Desempenho dos 6 Modelos no Test Set', fontsize=14, pad=15)
    plt.ylim(0, 1.05)
    plt.ylabel('Score')
    plt.xticks(rotation=15)
    plt.legend(loc='lower left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('metrics_comparison_6models.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    # 2. Matrizes de Confusão (Grid 2x3 para 6 modelos)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=class_names, yticklabels=class_names, cbar=False)
        axes[i].set_title(model_name, fontsize=12, pad=10)
        axes[i].set_ylabel('Rótulo Verdadeiro')
        axes[i].set_xlabel('Previsão do Modelo')
        
    plt.tight_layout()
    plt.savefig('confusion_matrices_6models.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    print("\nPronto! Gráficos salvos: 'metrics_comparison_6models.pdf' e 'confusion_matrices_6models.pdf'")


# =========================================================
# 5. EXECUÇÃO PRINCIPAL
# =========================================================
if __name__ == "__main__":
    SPLITS_CSV_PATH = "seu_arquivo_splits.csv" # Mude para o nome correto
    
    # Carregando dados
    df = pd.read_csv(SPLITS_CSV_PATH)
    train_df = df[df['split'] != 'test'].copy() 
    test_df = df[df['split'] == 'test'].copy()
    
    classes_names = sorted(['grotesco', 'serifado', 'escritural', 'fantasia'])
    
    # --- PREPARANDO LOADERS DE IMAGENS ---
    # Para testes e modelos sem augmentation
    loader_base_train = DataLoader(TypographyDataset(train_df, transform=base_transform), batch_size=32, shuffle=True)
    loader_test = DataLoader(TypographyDataset(test_df, transform=base_transform), batch_size=32, shuffle=False)
    
    # Para modelos com augmentation
    loader_aug_train = DataLoader(TypographyDataset(train_df, transform=train_transform_aug), batch_size=32, shuffle=True)
    
    # Dicionário que guardará todas as previsões
    model_preds = {}
    
    # --- 1 & 2. MACHINE LEARNING CLÁSSICO ---
    print("\n--- PREPARANDO MACHINE LEARNING CLÁSSICO ---")
    X_train, y_train = extract_classical_features(train_df, target_size=(64, 64))
    X_test, y_true_test = extract_classical_features(test_df, target_size=(64, 64))
    
    print("Treinando Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=0)
    rf.fit(X_train, y_train)
    model_preds['1. Random Forest'] = rf.predict(X_test)
    
    print("Treinando XGBoost...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    model_preds['2. XGBoost'] = xgb.predict(X_test)
    
    # --- 3. RESNET-18 (SEM AUG, 10 ÉPOCAS) ---
    print("\n--- INICIANDO DEEP LEARNING ---")
    res_base = train_dl_model('3. ResNet-18 (Base)', loader_base_train, epochs=10)
    model_preds['3. ResNet-18 (Base)'] = predict_dl_model(res_base, loader_test)
    
    # --- 4. RESNET-18 (COM AUG, 40 ÉPOCAS) ---
    res_aug = train_dl_model('4. ResNet-18 (Augmentation)', loader_aug_train, epochs=40)
    model_preds['4. ResNet-18 (Aug)'] = predict_dl_model(res_aug, loader_test)
    
    # --- 5. EFFICIENTNET-B0 (SEM AUG, 10 ÉPOCAS) ---
    eff_base = train_dl_model('5. EfficientNet (Base)', loader_base_train, epochs=10)
    model_preds['5. EfficientNet (Base)'] = predict_dl_model(eff_base, loader_test)
    
    # --- 6. EFFICIENTNET-B0 (COM AUG, 40 ÉPOCAS) ---
    # Assumi 40 épocas para igualar com a ResNet com augmentation
    eff_aug = train_dl_model('6. EfficientNet (Augmentation)', loader_aug_train, epochs=40)
    model_preds['6. EfficientNet (Aug)'] = predict_dl_model(eff_aug, loader_test)
    
    # --- CONFRONTO FINAL ---
    compare_models(y_true_test, model_preds, classes_names)