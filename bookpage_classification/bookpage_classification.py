import os
import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from PIL import Image
from torchvision import models, transforms

# ==========================================
# 1. CONFIGURAÇÕES E DICIONÁRIOS
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ['escritural', 'fantasia', 'grotesco', 'serifado']

# O esquema de cores definido
COLOR_MAP = {
    'grotesco': 'blue',
    'escritural': 'yellow',
    'serifado': 'red',
    'fantasia': 'green'
}

# Transformação idêntica à do treinamento
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. FUNÇÕES DE INFERÊNCIA
# ==========================================
def load_trained_model(model_path):
    import torch.nn as nn
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def predict_cropped_image(pil_img, model):
    """
    Recebe diretamente um objeto PIL Image da letra recortada na memória,
    sem precisar ler do disco.
    """
    img_tensor = preprocess(pil_img)
    img_batch = img_tensor.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_batch)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted_idx = torch.max(outputs, 1)
        
    class_idx = predicted_idx.item()
    predicted_class = CLASSES[class_idx]
    confidence = probabilities[class_idx].item() * 100
    
    return predicted_class, confidence

# ==========================================
# 3. PIPELINE PRINCIPAL DE PÁGINA
# ==========================================
def process_and_visualize_page(image_path, model):
    print(f"\nAnalisando a página: {image_path}")
    
    # 1. Carrega as imagens
    original_bgr = cv.imread(image_path)
    if original_bgr is None:
        print("Erro ao carregar a imagem.")
        return
        
    original_rgb = cv.cvtColor(original_bgr, cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(original_bgr, cv.COLOR_BGR2GRAY)
    img_h, img_w = original_rgb.shape[:2]
    
    # 2. Binarização simples (igual ao treino)
    _, binarized = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)

    # 3. Fechamento morfológico (igual ao treino — kernel 21x21)
    kernel = np.ones((11, 11), np.uint8)
    img_inverted = 255 - binarized
    closing_result = cv.morphologyEx(img_inverted, cv.MORPH_CLOSE, kernel)
    morph_closed = 255 - closing_result

    # 4. Rotulação por componentes conexas (igual ao treino — skimage)
    from skimage import measure
    labeled_img, num_labels = measure.label(morph_closed, connectivity=2, background=255, return_num=True)
    props = measure.regionprops(labeled_img)
    # 4. Configurar o Plot
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(original_rgb)
    ax.axis('off')
    
    count_valid_letters = 0
    
    # 5. Iterar sobre cada Bounding Box
    for region in props:
        bbox = region.bbox  # (min_row, min_col, max_row, max_col)
        y1, x1, y2, x2 = bbox[0], bbox[1], bbox[2], bbox[3]
        h = y2 - y1
        w = x2 - x1

        if w * h < 100 or h > img_h * 0.5:
            continue

        # Sem padding adicional — o fechamento morfológico já deu "respiro"
        # Se quiser padding, aplique aqui de forma fixa (ex: 5px fixos, não proporcional)
        cropped_letter_array = original_rgb[y1:y2, x1:x2]
        if cropped_letter_array.size == 0:
            continue

        pil_letter = Image.fromarray(cropped_letter_array)
        pred_class, confidence = predict_cropped_image(pil_letter, model)


        # 8. Desenhar o Bounding Box
        base_color = COLOR_MAP[pred_class]
        alpha_val = max(0.15, confidence / 100.0) # Garante que a caixa nunca suma 100% (mínimo 15% de opacidade)
        
        edge_color_with_alpha = mcolors.to_rgba(base_color, alpha=alpha_val)
        face_color_with_alpha = mcolors.to_rgba(base_color, alpha=alpha_val * 0.2)
        
        rect = patches.Rectangle(
            (x1, y1), 
            x2 - x1, 
            y2 - y1,
            linewidth=2, 
            edgecolor=edge_color_with_alpha, 
            facecolor=face_color_with_alpha
        )
        ax.add_patch(rect)
        
    print(f"Letras analisadas: {count_valid_letters}")
    
    # 9. Criar a Legenda
    legend_patches = []
    for cls_name, color in COLOR_MAP.items():
        patch = patches.Patch(color=color, label=cls_name.capitalize())
        legend_patches.append(patch)
        
    ax.legend(handles=legend_patches, loc='upper right', fontsize=12, framealpha=0.9)
    plt.title("Análise Tipográfica com Transparência por Confiança", fontsize=16)
    
    plt.tight_layout()
    plt.show()
    fig.savefig(f"resultado_{os.path.basename(image_path)}", dpi=300)

# ==========================================
# 4. TESTANDO O SCRIPT
# ==========================================
if __name__ == "__main__":
    # Certifique-se de que o caminho dos pesos do modelo está correto
    MODEL_PATH = "weights/resnet18_final.pth"
    
    if os.path.exists(MODEL_PATH):
        model = load_trained_model(MODEL_PATH)
        process_and_visualize_page("OR64_02.jpg", model)
    else:
        print(f"Pesos do modelo não encontrados em: {MODEL_PATH}")
