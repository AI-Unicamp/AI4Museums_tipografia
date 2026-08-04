import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# ==========================================
# CONFIGURAÇÕES DE DIRETÓRIO
# ==========================================
# Descomente e ajuste as montagens se estiver no Colab:
# from google.colab import drive
# drive.mount('/content/data', force_remount=True)

ROOT_DIRECTORY = '../../data'
IMAGES_DIRECTORY = ROOT_DIRECTORY + '/processed'
SPLITS_CSV_PATH = ROOT_DIRECTORY + '/k-fold_split/typography_dataset_splits.csv'

def prepare_grouped_dataset_splits(dataset_dir, n_splits=5, random_state=42):
    """
    Lê o dataset e divide TODO o conjunto em K folds, 
    garantindo que as gavetas fiquem juntas e as classes balanceadas.
    """
    dataset_path = Path(dataset_dir)
    data = []
    
    for class_dir in dataset_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        label = class_dir.name
        
        for sub_dir in class_dir.iterdir():
            if not sub_dir.is_dir():
                continue
                
            font_code = sub_dir.name
            drawer_id = f"{label}_{font_code}"
            
            for img_path in sub_dir.glob('*.*'): 
                if img_path.is_file() and img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    data.append({
                        'image_path': str(img_path),
                        'label': label,
                        'drawer_id': drawer_id
                    })
                    
    df = pd.DataFrame(data)
    
    if df.empty:
        raise ValueError("Nenhuma imagem encontrada. Verifique o caminho.")
        
    print(f"Total de imagens: {len(df)}")
    print(f"Total de gavetas únicas: {df['drawer_id'].nunique()}\n")
    
    # K-Fold Estratificado e Agrupado para todo o dataset
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Criamos uma coluna 'fold' inicializada com -1
    df['fold'] = -1
    
    print(f"--- Distribuindo em {n_splits} Folds ---")
    
    # O sgkf.split retorna train_idx e val_idx. 
    # O val_idx representa a partição não-sobreposta (o próprio fold).
    for fold_idx, (_, fold_test_idx) in enumerate(sgkf.split(df, df['label'], groups=df['drawer_id'])):
        df.loc[fold_test_idx, 'fold'] = fold_idx
        
        fold_data = df.iloc[fold_test_idx]
        print(f"Fold {fold_idx} -> "
              f"{len(fold_data)} imgs ({fold_data['drawer_id'].nunique()} gavetas)")
        
    return df

def save_splits_to_csv(df, output_filename="dataset_splits.csv"):
    """
    Salva o DataFrame consolidado no disco.
    """
    # Cria o diretório se não existir
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    df.to_csv(output_filename, index=False)
    print(f"\nSucesso! {len(df)} registros salvos em '{output_filename}'")
    return df

if __name__ == "__main__":
    if os.path.exists(IMAGES_DIRECTORY):
        # 1. Gera os splits totais
        df_splits = prepare_grouped_dataset_splits(
            dataset_dir=IMAGES_DIRECTORY, 
            n_splits=5 
        )
        
        # 2. Salva no disco
        save_splits_to_csv(df_splits, output_filename=SPLITS_CSV_PATH)
    else:
        print(f"Erro: Diretório {IMAGES_DIRECTORY} não encontrado.")