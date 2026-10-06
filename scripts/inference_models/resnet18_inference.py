"""
Inferência com o ResNet-18 treinado (resnet18_final_test.py). NÃO treina nada.

Classifica recortes de letras em: escritural, fantasia, grotesco, serifado
(a ordem/nomes reais vêm de resnet18_final_meta.json, gerado junto com os pesos).

Uso na linha de comando:
    python resnet18_inference.py letra_teste_A.png
    python resnet18_inference.py pasta_com_letras/ --csv predicoes.csv
    python resnet18_inference.py a.png b.png --weights weights/resnet18_final.pth
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# =========================================================
# CONFIGURAÇÕES
# =========================================================
DEFAULT_WEIGHTS = "weights/resnet18_final.pth"
# Usado apenas se o arquivo *_meta.json não existir ao lado dos pesos:
FALLBACK_CLASSES = ['escritural', 'fantasia', 'grotesco', 'serifado']
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Idêntico ao base_transform do treino
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# =========================================================
# MODELO
# =========================================================
def _load_class_names(weights_path):
    meta_path = weights_path.with_name(weights_path.stem + "_meta.json")
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)["class_names"]

    print(f"[aviso] {meta_path.name} não encontrado; usando a ordem padrão {FALLBACK_CLASSES}. "
          f"Confirme que ela é a mesma do treino.", file=sys.stderr)
    return list(FALLBACK_CLASSES)


def load_trained_model(weights_path=DEFAULT_WEIGHTS):
    """Recria o esqueleto do ResNet-18, injeta os pesos e devolve (modelo, nomes_das_classes)."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Pesos não encontrados em: {weights_path.resolve()}")

    class_names = _load_class_names(weights_path)

    model = models.resnet18(weights=None)  # sem baixar ImageNet: os pesos treinados já vêm no arquivo
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model, class_names


# =========================================================
# PREDIÇÃO
# =========================================================
def to_model_input(img):
    """
    O treino usou recortes lidos em tons de cinza (cv.IMREAD_GRAYSCALE) e replicados em 3 canais.
    Forçar o mesmo aqui não muda nada para imagens já cinzas e remove o tom do papel em recortes
    coloridos de páginas, alinhando a inferência ao que o modelo viu no treino.
    """
    return preprocess(img.convert("L").convert("RGB"))


@torch.no_grad()
def predict_pil_batch(images, model, class_names):
    """Classifica uma lista de imagens PIL. Devolve uma lista de dicts (na mesma ordem)."""
    batch = torch.stack([to_model_input(im) for im in images]).to(device)
    probs = torch.softmax(model(batch), dim=1).cpu()

    results = []
    for p in probs:
        idx = int(p.argmax())
        results.append({
            "predicted": class_names[idx],
            "confidence": float(p[idx]) * 100,
            "probabilities": {c: float(p[i]) * 100 for i, c in enumerate(class_names)},
        })
    return results


def predict_pil(img, model, class_names):
    """Classifica uma única imagem PIL (por exemplo, um recorte de letra em memória)."""
    return predict_pil_batch([img], model, class_names)[0]


def predict_paths(paths, model, class_names, batch_size=64):
    """Classifica arquivos de imagem. Arquivos ilegíveis são ignorados com aviso."""
    rows = []
    for start in range(0, len(paths), batch_size):
        images, valid_paths = [], []
        for p in paths[start:start + batch_size]:
            try:
                images.append(Image.open(p).convert("L"))
                valid_paths.append(p)
            except Exception as e:
                print(f"[aviso] não foi possível abrir {p}: {e}", file=sys.stderr)

        if not images:
            continue

        for p, res in zip(valid_paths, predict_pil_batch(images, model, class_names)):
            rows.append({"path": str(p), **res})
    return rows


# =========================================================
# LINHA DE COMANDO
# =========================================================
def collect_image_paths(inputs):
    paths = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"[aviso] não encontrado: {p}", file=sys.stderr)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Classificação tipográfica de letras com ResNet-18 (sem treino).")
    parser.add_argument("inputs", nargs="+", help="Imagens e/ou pastas com imagens (busca recursiva).")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help=f"Arquivo .pth (padrão: {DEFAULT_WEIGHTS}).")
    parser.add_argument("--csv", default=None, help="Se informado, salva as predições neste CSV.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    paths = collect_image_paths(args.inputs)
    if not paths:
        sys.exit("Nenhuma imagem encontrada.")

    print(f"Usando dispositivo para inferência: {device}")
    model, class_names = load_trained_model(args.weights)
    rows = predict_paths(paths, model, class_names, args.batch_size)

    for r in rows:
        probs = " | ".join(f"{c} {v:.1f}%" for c, v in r["probabilities"].items())
        print(f"{Path(r['path']).name}: {r['predicted'].upper()} ({r['confidence']:.1f}%)  ->  {probs}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "predicted", "confidence"] + [f"prob_{c}" for c in class_names])
            for r in rows:
                writer.writerow([r["path"], r["predicted"], f"{r['confidence']:.2f}"] +
                                [f"{r['probabilities'][c]:.2f}" for c in class_names])
        print(f"\nPredições salvas em: {args.csv}")


if __name__ == "__main__":
    main()
