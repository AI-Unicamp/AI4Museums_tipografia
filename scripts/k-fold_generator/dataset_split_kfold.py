import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# ==========================================
# CONFIGURAÇÕES DE DIRETÓRIO
# ==========================================

ROOT_DIRECTORY = 'data'
IMAGES_DIRECTORY = ROOT_DIRECTORY + '/processed'
SPLITS_CSV_PATH = ROOT_DIRECTORY + '/k-fold_split/typography_dataset_splits.csv'


def load_image_index(dataset_dir):
    """
    Varre <dataset_dir>/<classe>/<codigo_da_gaveta>/*.png|jpg|jpeg
    e devolve um DataFrame com image_path, label e drawer_id.
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
    return df


def _assign_folds(df, n_splits, seed):
    """Uma tentativa de StratifiedGroupKFold com uma semente específica."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1, dtype=int)
    for fold_idx, (_, fold_test_idx) in enumerate(
        sgkf.split(df, df['label'], groups=df['drawer_id'])
    ):
        fold[fold_test_idx] = fold_idx
    return fold


def _imbalance_score(counts, class_totals, n_splits):
    """
    Quanto menor, mais parecida com a proporção global é a distribuição
    de cada classe entre os folds (desvio relativo quadrático somado).
    `counts`: DataFrame (folds x classes) com a contagem de imagens.
    """
    expected = class_totals / n_splits
    return float((((counts - expected) / expected) ** 2).sum().sum())


def prepare_grouped_dataset_splits(dataset_dir, n_splits=5, random_state=42,
                                   min_per_class_per_fold=1, max_attempts=500):
    """
    Divide todo o conjunto em K folds com:
      - agrupamento por gaveta (a mesma gaveta nunca aparece em dois folds);
      - estratificação por classe (StratifiedGroupKFold);
      - garantia de pelo menos `min_per_class_per_fold` imagens de CADA classe
        (em particular 'fantasia') em todos os folds.

    O StratifiedGroupKFold é uma heurística gulosa e NÃO garante o mínimo por
    classe sozinho. Por isso tentamos várias sementes, descartamos as divisões
    que deixam alguma classe sem suporte em algum fold e ficamos com a mais
    balanceada entre as válidas.
    """
    df = load_image_index(dataset_dir)

    print(f"Total de imagens: {len(df)}")
    print(f"Total de gavetas únicas: {df['drawer_id'].nunique()}\n")

    # ----- Checagem de viabilidade -----
    drawers_per_class = df.groupby('label')['drawer_id'].nunique()
    images_per_class = df['label'].value_counts()
    print("Imagens e gavetas por classe:")
    print(pd.DataFrame({'imagens': images_per_class, 'gavetas': drawers_per_class}).to_string())
    print()

    needed_drawers = n_splits * min_per_class_per_fold
    too_few = drawers_per_class[drawers_per_class < needed_drawers]
    if not too_few.empty:
        raise ValueError(
            f"Inviável com agrupamento por gaveta: as classes {too_few.to_dict()} têm menos "
            f"gavetas do que o necessário ({needed_drawers} = {n_splits} folds x "
            f"{min_per_class_per_fold}). Opções: reduzir n_splits, ou relaxar o agrupamento "
            f"para essa classe (o que reintroduz risco de vazamento entre treino e teste)."
        )

    # ----- Busca da melhor divisão válida -----
    class_totals = images_per_class.sort_index()
    best_fold, best_score, best_seed = None, np.inf, None
    n_valid = 0

    for attempt in range(max_attempts):
        seed = random_state + attempt
        fold = _assign_folds(df, n_splits, seed)

        counts = pd.crosstab(fold, df['label']).reindex(columns=class_totals.index, fill_value=0)
        if len(counts) != n_splits or counts.values.min() < min_per_class_per_fold:
            continue

        n_valid += 1
        score = _imbalance_score(counts, class_totals, n_splits)
        if score < best_score:
            best_fold, best_score, best_seed = fold, score, seed

    if best_fold is None:
        raise RuntimeError(
            f"Nenhuma das {max_attempts} tentativas produziu folds com >= "
            f"{min_per_class_per_fold} imagem(ns) de cada classe. Aumente max_attempts "
            f"ou reduza n_splits."
        )

    df['fold'] = best_fold
    print(f"{n_valid}/{max_attempts} tentativas válidas. "
          f"Escolhida a semente {best_seed} (score de desbalanceamento = {best_score:.4f}).\n")

    # ----- Relatório -----
    print(f"--- Distribuição final em {n_splits} folds ---")
    for fold_idx in range(n_splits):
        fold_data = df[df['fold'] == fold_idx]
        print(f"Fold {fold_idx} -> {len(fold_data)} imgs ({fold_data['drawer_id'].nunique()} gavetas)")

    print("\nImagens por classe em cada fold:")
    print(pd.crosstab(df['fold'], df['label']).to_string())

    # Verificação de segurança: nenhuma gaveta pode estar em mais de um fold
    drawers_in_multiple_folds = df.groupby('drawer_id')['fold'].nunique()
    assert (drawers_in_multiple_folds == 1).all(), "Vazamento: gaveta presente em mais de um fold!"

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
            n_splits=5,
            min_per_class_per_fold=1
        )

        # 2. Salva no disco
        save_splits_to_csv(df_splits, output_filename=SPLITS_CSV_PATH)
    else:
        print(f"Erro: Diretório {IMAGES_DIRECTORY} não encontrado.")