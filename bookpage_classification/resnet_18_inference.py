import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==========================================
# 1. CONFIGURAÇÕES INICIAIS
# ==========================================
# Mapeamento exato de como o LabelEncoder organizou (ordem alfabética)
CLASSES = ['escritural', 'fantasia', 'grotesco', 'serifado']

# Caminho do modelo salvo no script anterior
MODEL_PATH = "weights/resnet18_final.pth" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo para inferência: {device}")

# ==========================================
# 2. TRANSFORMAÇÃO DA IMAGEM
# ==========================================
# IMPORTANTE: Deve ser matematicamente idêntico ao 'base_transform' do treino
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. CARREGAMENTO DO MODELO
# ==========================================
def load_trained_model(model_path):
    # 1. Recria o esqueleto da ResNet-18
    # (Não precisamos baixar os pesos do ImageNet de novo, pois vamos usar os seus)
    model = models.resnet18(weights=None) 
    
    # 2. Ajusta a última camada para 4 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    
    # 3. Injeta os pesos treinados
    # (map_location garante que funcione mesmo se você treinou na GPU e testar na CPU)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 4. Envia pro dispositivo e coloca em modo de avaliação
    model = model.to(device)
    model.eval()
    
    return model

# ==========================================
# 4. FUNÇÃO DE PREDIÇÃO
# ==========================================
def predict_image(image_path, model):
    # Abre a imagem e garante que está em RGB (evita erro com imagens em tons de cinza puras ou PNGs com transparência)
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Erro ao abrir a imagem: {e}")
        return None
    
    # Aplica o redimensionamento e a normalização
    img_tensor = preprocess(img)
    
    # O PyTorch espera um "lote" (batch) de imagens, não uma só.
    # O unsqueeze(0) transforma a imagem de (3, 224, 224) para (1, 3, 224, 224)
    img_batch = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Passa a imagem pela rede
        outputs = model(img_batch)
        
        # Converte a saída bruta (logits) em probabilidades de 0 a 100%
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Pega o índice da classe com a maior probabilidade
        _, predicted_idx = torch.max(outputs, 1)
        
    class_idx = predicted_idx.item()
    predicted_class = CLASSES[class_idx]
    confidence = probabilities[class_idx].item() * 100
    
    # Retorna também o dicionário com as probabilidades de todas as classes, caso queira analisar a dúvida do modelo
    all_probs = {CLASSES[i]: probs.item() * 100 for i, probs in enumerate(probabilities)}
    
    return predicted_class, confidence, all_probs

# ==========================================
# 5. TESTANDO NA PRÁTICA
# ==========================================
if __name__ == "__main__":
    # Carrega o modelo uma única vez
    meu_modelo = load_trained_model(MODEL_PATH)
    
    # Coloque o caminho de qualquer imagem aqui
    imagem_teste = "letra_teste_L.png"
    
    print(f"\nAnalisando a imagem: {imagem_teste}")
    resultado, confianca, detalhes = predict_image(imagem_teste, meu_modelo)
    
    print("\n" + "="*40)
    print(f"RESULTADO FINAL: {resultado.upper()}")
    print(f"CONFIANÇA: {confianca:.2f}%")
    print("="*40)
    
    print("\nProbabilidades por classe:")
    for classe, prob in detalhes.items():
        print(f"- {classe.capitalize()}: {prob:.2f}%")

