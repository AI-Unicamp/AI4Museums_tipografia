import os
import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from skimage import measure
from PIL import Image
from torchvision import models, transforms

# ==========================================
# 1. CONFIGURAÇÕES E DICIONÁRIOS
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ['escritural', 'fantasia', 'grotesco', 'serifado']

# O esquema de cores que você definiu
COLOR_MAP = {
    'escritural': 'yellow',
    'grotesco': 'blue',
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
    
    # 1. Carrega as imagens (RGB para o plot final/ResNet, e Grayscale para o OpenCV)
    original_bgr = cv.imread(image_path)
    if original_bgr is None:
        print("Erro ao carregar a imagem.")
        return
        
    original_rgb = cv.cvtColor(original_bgr, cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(original_bgr, cv.COLOR_BGR2GRAY)
    
    # 2. Pré-processamento (Filtro e Binarização)
    blur = cv.medianBlur(img_gray, 15)
    _, binarized = cv.threshold(blur, 127, 255, cv.THRESH_BINARY)
    
    # 3. Fechamento Morfológico
    img_inverted = 255 - binarized
    kernel = np.ones((7, 7), np.uint8)
    closing_result = cv.morphologyEx(img_inverted, cv.MORPH_CLOSE, kernel)
    final_mask = 255 - closing_result
    
    # 4. Encontrar Componentes Conectados
    labeled_img, num_labels = measure.label(final_mask, connectivity=2, background=255, return_num=True)
    props = measure.regionprops(labeled_img)
    
    # 5. Configurar o Plot
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(original_rgb)
    ax.axis('off')
    
    count_valid_letters = 0
    
    # 6. Iterar sobre cada Bounding Box
    for i in range(num_labels):
        bbox = props[i].bbox
        min_row, min_col, max_row, max_col = bbox
        
        # Ignora caixas muito pequenas
        if (max_row - min_row) * (max_col - min_col) < 100:
            continue
            
        count_valid_letters += 1
        
        # Recorta a letra da imagem RGB original para enviar à ResNet
        cropped_letter_array = original_rgb[min_row:max_row, min_col:max_col]
        pil_letter = Image.fromarray(cropped_letter_array)
        
        # 7. Inferência
        pred_class, confidence = predict_cropped_image(pil_letter, model)
        
        # 8. Desenhar o Bounding Box
        base_color = COLOR_MAP[pred_class]
        alpha_val = max(0.15, confidence / 100.0) # Garante que a caixa nunca suma 100% (mínimo 15% de opacidade)
        
        # Cria a cor com o canal alpha aplicado (transparência baseada na confiança)
        edge_color_with_alpha = mcolors.to_rgba(base_color, alpha=alpha_val)
        
        # Uma sacada legal: preencher a caixa com a mesma cor, mas bem mais transparente
        face_color_with_alpha = mcolors.to_rgba(base_color, alpha=alpha_val * 0.2)
        
        rect = patches.Rectangle(
            (min_col, min_row), 
            max_col - min_col, 
            max_row - min_row,
            linewidth=2, 
            edgecolor=edge_color_with_alpha, 
            facecolor=face_color_with_alpha
        )
        ax.add_patch(rect)
        
    print(f"Letras analisadas: {count_valid_letters}")
    
    # 9. Criar a Legenda
    legend_patches = []
    for cls_name, color in COLOR_MAP.items():
        # Cria um retângulo dummy só para a legenda
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
    MODEL_PATH = "../scripts/weights/resnet18_final.pth"
    model = load_trained_model(MODEL_PATH)
    
    process_and_visualize_page("OR64_02.jpg", model)
    pass