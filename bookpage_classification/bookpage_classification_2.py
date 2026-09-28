import os
import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from PIL import Image
from torchvision import models, transforms
from skimage import measure
from collections import Counter

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['escritural', 'fantasia', 'grotesco', 'serifado']
COLOR_MAP = {
    'escritural': 'yellow',
    'grotesco':   'blue',
    'serifado':   'red',
    'fantasia':   'green'
}

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. MODELO
# ==========================================
def load_trained_model(model_path):
    import torch.nn as nn
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def predict_cropped_image(pil_img, model):
    img_tensor = preprocess(pil_img)
    img_batch = img_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_batch)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted_idx = torch.max(outputs, 1)
    class_idx = predicted_idx.item()
    return CLASSES[class_idx], probs[class_idx].item() * 100

# ==========================================
# 3. SEGMENTAÇÃO EM DOIS NÍVEIS
# ==========================================
def get_bboxes(binary_inv, kernel_size):
    """Fecha morfologicamente e retorna bboxes dos componentes conexos."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv.morphologyEx(binary_inv, cv.MORPH_CLOSE, kernel)
    labeled, num_labels = measure.label(255 - closed, connectivity=2,
                                        background=255, return_num=True)
    props = measure.regionprops(labeled)
    return props

def classify_word_region(word_bbox, binary_inv, original_rgb, model, img_h, img_w,
                          kernel_letter=15, min_letter_px=100):
    """
    Dado o bbox de uma palavra (do kernel 13x13), recorta essa região,
    segmenta letras individuais dentro dela (kernel 15x15) e vota.
    Retorna (classe_vencedora, confiança_média, lista_de_bboxes_de_letras_em_coords_globais).
    """
    wy1, wx1, wy2, wx2 = word_bbox

    # Recorte da região da palavra na imagem binarizada invertida
    region_inv = binary_inv[wy1:wy2, wx1:wx2]
    region_rgb = original_rgb[wy1:wy2, wx1:wx2]

    # Segmenta letras dentro da palavra com kernel maior
    kernel = np.ones((kernel_letter, kernel_letter), np.uint8)
    closed = cv.morphologyEx(region_inv, cv.MORPH_CLOSE, kernel)
    labeled, _ = measure.label(255 - closed, connectivity=2,
                                background=255, return_num=True)
    letter_props = measure.regionprops(labeled)

    votes = []
    confidences = []
    letter_bboxes_global = []

    for lp in letter_props:
        ly1, lx1, ly2, lx2 = lp.bbox
        lh = ly2 - ly1
        lw = lx2 - lx1

        if lw * lh < min_letter_px or lh > (wy2 - wy1) * 0.95:
            continue

        # Coordenadas globais para desenhar depois
        gy1, gx1 = wy1 + ly1, wx1 + lx1
        gy2, gx2 = wy1 + ly2, wx1 + lx2
        letter_bboxes_global.append((gx1, gy1, gx2, gy2))

        crop = original_rgb[gy1:gy2, gx1:gx2]
        if crop.size == 0:
            continue

        pil_letter = Image.fromarray(crop)
        pred_class, confidence = predict_cropped_image(pil_letter, model)
        votes.append(pred_class)
        confidences.append(confidence)

    if not votes:
        return None, None, []

    # Votação por maioria
    winner = Counter(votes).most_common(1)[0][0]
    avg_conf = np.mean(confidences)

    return winner, avg_conf, letter_bboxes_global

# ==========================================
# 4. PIPELINE PRINCIPAL
# ==========================================
def process_and_visualize_page(image_path, model,
                                kernel_word=13, kernel_letter=15,
                                min_word_px=500, max_word_height_ratio=0.5):

    print(f"\nAnalisando: {image_path}")
    original_bgr = cv.imread(image_path)
    if original_bgr is None:
        print("Erro ao carregar a imagem.")
        return

    original_rgb = cv.cvtColor(original_bgr, cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(original_bgr, cv.COLOR_BGR2GRAY)
    img_h, img_w = original_rgb.shape[:2]

    # Binarização e inversão (igual ao treino)
    _, binarized = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
    binary_inv = 255 - binarized

    # Nível 1: bboxes de palavras
    word_props = get_bboxes(binary_inv, kernel_word)

    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(original_rgb)
    ax.axis('off')

    n_words = 0
    n_letters = 0

    for wp in word_props:
        wy1, wx1, wy2, wx2 = wp.bbox
        wh = wy2 - wy1
        ww = wx2 - wx1

        if ww * wh < min_word_px or wh > img_h * max_word_height_ratio:
            continue

        # Nível 2: classifica letras dentro da palavra e vota
        winner, avg_conf, letter_bboxes = classify_word_region(
            (wy1, wx1, wy2, wx2), binary_inv, original_rgb, model, img_h, img_w,
            kernel_letter=kernel_letter
        )

        if winner is None:
            continue

        n_words += 1
        n_letters += len(letter_bboxes)

        base_color = COLOR_MAP[winner]
        alpha_val = max(0.15, avg_conf / 100.0)
        edge_rgba = mcolors.to_rgba(base_color, alpha=alpha_val)
        face_rgba = mcolors.to_rgba(base_color, alpha=alpha_val * 0.15)

        # Desenha bbox da PALAVRA
        rect_word = patches.Rectangle(
            (wx1, wy1), ww, wh,
            linewidth=2.5, edgecolor=edge_rgba, facecolor=face_rgba
        )
        ax.add_patch(rect_word)

        # Opcional: desenha bboxes das LETRAS individuais (mais fino, mais transparente)
        for (lx1, ly1, lx2, ly2) in letter_bboxes:
            rect_letter = patches.Rectangle(
                (lx1, ly1), lx2 - lx1, ly2 - ly1,
                linewidth=0.8,
                edgecolor=mcolors.to_rgba(base_color, alpha=alpha_val * 0.5),
                facecolor='none'
            )
            ax.add_patch(rect_letter)

    print(f"Palavras classificadas: {n_words}")
    print(f"Letras individuais analisadas: {n_letters}")

    # Legenda
    legend_patches = [patches.Patch(color=c, label=cls.capitalize())
                      for cls, c in COLOR_MAP.items()]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=12, framealpha=0.9)
    plt.title("Análise Tipográfica — Segmentação em Dois Níveis", fontsize=16)
    plt.tight_layout()

    out_path = f"resultado_{os.path.basename(image_path)}"
    fig.savefig(out_path, dpi=300)
    plt.show()
    print(f"Salvo em: {out_path}")

# ==========================================
# 5. EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    MODEL_PATH = "weights/resnet18_final.pth"
    if os.path.exists(MODEL_PATH):
        model = load_trained_model(MODEL_PATH)
        process_and_visualize_page("OR64_02.jpg", model)
    else:
        print(f"Pesos não encontrados em: {MODEL_PATH}")