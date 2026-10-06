#!/usr/bin/env python3
"""
Segmentação de letras - Parte 1 (conjunto data_1)

Imagens de dimensões grandes, porém em condições adversas de captura: pouco
contraste entre letra e fundo, fundo escuro, iluminação ruim etc.

Abordagem: componentes conexas (vizinhança-8).

Pipeline:
    1. Pré-processamento
        1.1 Binarização simples        -> <root>/interim_data_1/binarized
        1.2 Fechamento morfológico     -> <root>/interim_data_1/morph_closed
    2. Segmentação por componentes conexas. Os limites (bounding boxes) são
       obtidos nas imagens transformadas, mas os recortes são feitos sobre as
       imagens ORIGINAIS (<root>/raw_data_1)
       -> <root>/processed_data_1

Estrutura de pastas esperada (cada subpasta de `raw_data_1` é uma família/estilo):
    <root>/raw_data_1/<estilo>/<cod_XXX>.(png|jpg|jpeg)
As pastas interim_data_1 e processed_data_1 são criadas pelo script.

Uso (a partir da raiz do repositório):
    python scripts/preprocessing_and_segmentation/crop_data_1.py
    python scripts/preprocessing_and_segmentation/crop_data_1.py --root caminho/para/data_1
"""

import argparse
import os

import cv2 as cv
import numpy as np
from skimage import measure

# --------------------------------------------------------------------------
# Parâmetros
# --------------------------------------------------------------------------

# Extensões de imagem aceitas
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# Binarização: valores <= BINARY_THRESHOLD viram 0 (preto, idealmente as
# letras); valores > BINARY_THRESHOLD viram 255 (branco, idealmente o fundo)
BINARY_THRESHOLD = 127

# Filtro de mediana (remoção de ruído "sal e pimenta"). ATENÇÃO: no notebook
# original o filtro era calculado, mas seu resultado nunca era usado (a
# binarização era aplicada sobre a imagem sem filtrar). Por isso o padrão aqui
# é False, para reproduzir exatamente os resultados originais. Mude para True
# se quiser que o filtro realmente seja aplicado antes da binarização.
USE_MEDIAN_BLUR = False
MEDIAN_KERNEL_SIZE = 15

# Fechamento morfológico: junta pedaços de uma mesma letra que ficaram
# desconectados após a binarização (falhas de impressão, iluminação não
# uniforme etc.). Em imagens grandes, usa-se um kernel grande (21x21).
MORPH_KERNEL_SIZE = 21

# Recortes com menos pixels que isso são descartados (ruído remanescente)
MIN_CROP_PIXELS = 100


# --------------------------------------------------------------------------
# Utilitários
# --------------------------------------------------------------------------

def list_images(root_folder):
    """
    Percorre `root_folder` recursivamente e devolve uma lista de tuplas
    (caminho_da_imagem, nome_do_estilo, nome_base_da_imagem).

    O "estilo" é o nome da pasta que contém a imagem (ex.: escritural,
    fantasia...). O "nome base" é o nome do arquivo sem extensão (ex.: cod_002).
    """
    images = []
    for current_path, _, files in os.walk(root_folder):
        for filename in sorted(files):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue
            images.append((
                os.path.join(current_path, filename),
                os.path.basename(current_path),
                os.path.splitext(filename)[0],
            ))
    return images


# --------------------------------------------------------------------------
# Etapa 1.1: binarização
# --------------------------------------------------------------------------

def binarize_images(raw_folder, output_folder):
    """
    Lê as imagens originais em escala de cinza, binariza com limiar simples e
    salva em `output_folder/<estilo>/<nome>.png`.
    """
    for input_path, style_name, base_name in list_images(raw_folder):
        output_dir = os.path.join(output_folder, style_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"[binarização] {input_path}")

        img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            print("  -> Erro ao carregar a imagem. Ignorando...")
            continue

        if USE_MEDIAN_BLUR:
            img = cv.medianBlur(img, MEDIAN_KERNEL_SIZE)

        _, binarized = cv.threshold(img, BINARY_THRESHOLD, 255, cv.THRESH_BINARY)

        cv.imwrite(os.path.join(output_dir, f"{base_name}.png"), binarized)


# --------------------------------------------------------------------------
# Etapa 1.2: fechamento morfológico
# --------------------------------------------------------------------------

def morphological_closing(binarized_folder, output_folder):
    """
    Aplica fechamento morfológico às imagens binarizadas e salva em
    `output_folder/<estilo>/<nome>.png`.

    Detalhes dos resultados (ex.: perda de detalhes das letras) não importam:
    só nos interessam os bounding boxes dos caracteres.
    """
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)

    for input_path, style_name, base_name in list_images(binarized_folder):
        output_dir = os.path.join(output_folder, style_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"[fechamento morfológico] {input_path}")

        img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            print("  -> Erro ao carregar a imagem. Ignorando...")
            continue

        # As operações morfológicas tratam o branco (255) como "objeto".
        # Como as letras são pretas, inverte-se a imagem antes do fechamento
        # e inverte-se novamente depois.
        img_inverted = 255 - img
        closing_result = cv.morphologyEx(img_inverted, cv.MORPH_CLOSE, kernel)
        result = 255 - closing_result

        cv.imwrite(os.path.join(output_dir, f"{base_name}.png"), result)


# --------------------------------------------------------------------------
# Etapa 2: segmentação por componentes conexas
# --------------------------------------------------------------------------

def segment_letters(raw_folder, aux_folder, output_folder):
    """
    Encontra as componentes conexas (vizinhança-8) nas imagens transformadas de
    `aux_folder` (morph_closed) e recorta, a partir das imagens ORIGINAIS de
    `raw_folder`, o bounding box de cada componente.

    Os recortes são salvos em:
        output_folder/<estilo>/<nome_da_imagem>/<nome_da_imagem>_<i>.png
    """
    for input_path, style_name, base_name in list_images(raw_folder):
        output_dir = os.path.join(output_folder, style_name, base_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"[segmentação] {input_path}")
        print(f"  -> estilo: '{style_name}', código: '{base_name}'")
        print(f"  -> recortes serão salvos em: {output_dir}")

        # Imagem original (em escala de cinza): dela saem os recortes
        original_img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
        if original_img is None:
            print("  -> Erro ao carregar a imagem original. Ignorando...")
            continue

        # Imagem transformada (pré-processada): usada só para achar as letras.
        # Sem IMREAD_GRAYSCALE, é carregada com 3 canais (BGR).
        aux_path = os.path.join(aux_folder, style_name, f"{base_name}.png")
        img = cv.imread(aux_path)
        if img is None:
            print(f"  -> Imagem pré-processada não encontrada: {aux_path}. Ignorando...")
            continue

        # measure.label rotula cada componente conexa (0 ou rótulo do fundo
        # fica de fora; 1, 2, 3... para cada componente). `background=255`
        # indica que o fundo é branco. `connectivity=2` em imagem 2D equivale
        # à vizinhança-8 (cima, baixo, esquerda, direita e diagonais).
        labeled_img, num_labels = measure.label(
            img, connectivity=2, background=255, return_num=True
        )

        # regionprops extrai propriedades de cada componente, inclusive o bbox
        props = measure.regionprops(labeled_img)

        for i in range(num_labels):
            # Como `img` tem 3 dimensões (altura, largura, canais), o bbox é
            # (min_linha, min_coluna, min_canal, max_linha, max_coluna,
            # max_canal). Por isso usamos os índices 0, 1, 3 e 4.
            bbox = props[i].bbox
            cropped_letter = original_img[bbox[0]:bbox[3], bbox[1]:bbox[4]]

            # Descarta recortes muito pequenos (ruído remanescente)
            if np.size(cropped_letter) < MIN_CROP_PIXELS:
                continue

            output_path = os.path.join(output_dir, f"{base_name}_{i:03d}.png")
            cv.imwrite(output_path, cropped_letter)


# --------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pré-processamento e segmentação de letras (conjunto data_1)."
    )
    parser.add_argument(
        "--root",
        default="data/interim/data_1",
        help="Pasta raiz do conjunto, contendo a subpasta 'raw_data_1' "
        "(padrão: data/interim/data_1).",
    )
    args = parser.parse_args()

    # Os nomes das subpastas seguem o nome do conjunto (ex.: raw_data_1)
    name = os.path.basename(os.path.normpath(args.root))
    raw_folder = os.path.join(args.root, f"raw_{name}")
    interim_folder = os.path.join(args.root, f"interim_{name}")
    binarized_folder = os.path.join(interim_folder, "binarized")
    closed_folder = os.path.join(interim_folder, "morph_closed")
    processed_folder = os.path.join(args.root, f"processed_{name}")

    if not os.path.isdir(raw_folder):
        raise SystemExit(f"Pasta de imagens originais não encontrada: {raw_folder}")

    # Etapa 1: pré-processamento
    binarize_images(raw_folder, binarized_folder)
    morphological_closing(binarized_folder, closed_folder)

    # Etapa 2: segmentação (recortes a partir das imagens originais)
    segment_letters(raw_folder, closed_folder, processed_folder)

    print("\nConcluído.")


if __name__ == "__main__":
    main()
