#!/usr/bin/env python3
"""
Une os recortes de caracteres dos dois conjuntos em uma única pasta.

Origens (geradas por crop_data_1.py e crop_data_2.py):
    data/interim/data_1/processed_data_1/<estilo>/<cod_XXX>/<cod_XXX_NNN>.png
    data/interim/data_2/processed_data_2/<estilo>/<cod_XXX>/<cod_XXX_NNN>.png

Destino:
    data/interim/data_1_2_final/<estilo>/<cod_XXX>/<cod_XXX_NNN>.png

Obs.: o destino é intermediário. A pasta data/processed é reservada ao
dataset final de treinamento dos modelos e NÃO é alterada por este script.

A estrutura de subpastas é preservada; pastas de estilo com o mesmo nome nos
dois conjuntos (ex.: escritural) são mescladas.

Por padrão os arquivos são COPIADOS (as origens não são alteradas). O script
pode ser executado mais de uma vez: arquivos que já existem no destino com
conteúdo idêntico são ignorados. Se um arquivo de mesmo caminho existir com
conteúdo DIFERENTE, o script aborta sem copiar nada, a menos que se use
--overwrite.

Uso (a partir da raiz do repositório):
    python scripts/preprocessing_and_segmentation/merge_processed.py
    python scripts/preprocessing_and_segmentation/merge_processed.py --move
    python scripts/preprocessing_and_segmentation/merge_processed.py --overwrite
"""

import argparse
import filecmp
import shutil
from pathlib import Path

# Pastas de origem e destino padrão (relativas ao diretório de execução)
DEFAULT_SOURCES = [
    "data/interim/data_1/processed_data_1",
    "data/interim/data_2/processed_data_2",
]
DEFAULT_OUTPUT = "data/interim/data_1_2_final"


def list_files(source):
    """
    Devolve (caminho_na_origem, caminho_relativo) para todo arquivo dentro de
    `source`, em qualquer nível de subpasta.
    """
    return [(p, p.relative_to(source)) for p in sorted(source.rglob("*")) if p.is_file()]


def main():
    parser = argparse.ArgumentParser(
        description="Une as pastas processed_data_1 e processed_data_2 em data/interim/data_1_2_final."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=DEFAULT_SOURCES,
        help="Pastas de origem (padrão: processed_data_1 e processed_data_2).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Pasta de destino (padrão: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move os arquivos em vez de copiá-los (remove as pastas de origem).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescreve arquivos do destino que tenham o mesmo caminho e conteúdo diferente.",
    )
    args = parser.parse_args()

    output = Path(args.output)

    # ----------------------------------------------------------------------
    # 1. Levanta tudo o que precisa ser transferido
    # ----------------------------------------------------------------------
    to_transfer = []   # (origem, destino)
    already_there = 0  # já existem no destino com conteúdo idêntico
    conflicts = []     # existem no destino com conteúdo diferente

    for source_arg in args.sources:
        source = Path(source_arg)
        if not source.is_dir():
            raise SystemExit(f"Pasta de origem não encontrada: {source}")

        for src_file, relative in list_files(source):
            dst_file = output / relative

            if dst_file.exists():
                if filecmp.cmp(src_file, dst_file, shallow=False):
                    already_there += 1
                    continue
                conflicts.append((src_file, dst_file))
                if not args.overwrite:
                    continue

            to_transfer.append((src_file, dst_file))

    # Dois arquivos de origem diferentes apontando para o mesmo destino
    # (ex.: mesmo cod_XXX nos dois conjuntos) também são conflito.
    seen = {}
    for src_file, dst_file in to_transfer:
        if dst_file in seen and not filecmp.cmp(seen[dst_file], src_file, shallow=False):
            conflicts.append((src_file, dst_file))
        seen.setdefault(dst_file, src_file)

    # ----------------------------------------------------------------------
    # 2. Aborta se houver conflitos (sem --overwrite), antes de copiar qualquer coisa
    # ----------------------------------------------------------------------
    if conflicts and not args.overwrite:
        print(f"{len(conflicts)} conflito(s): mesmo caminho de destino, conteúdo diferente.")
        for src_file, dst_file in conflicts[:10]:
            print(f"  {src_file}  ->  {dst_file}")
        if len(conflicts) > 10:
            print(f"  ... e mais {len(conflicts) - 10}")
        raise SystemExit("Nada foi copiado. Use --overwrite para sobrescrever.")

    # ----------------------------------------------------------------------
    # 3. Copia (ou move) os arquivos
    # ----------------------------------------------------------------------
    action = shutil.move if args.move else shutil.copy2
    for src_file, dst_file in to_transfer:
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        action(str(src_file), str(dst_file))

    verb = "movidos" if args.move else "copiados"
    print(f"{len(to_transfer)} arquivo(s) {verb} para {output}.")
    if already_there:
        print(f"{already_there} arquivo(s) já estavam no destino (idênticos) e foram ignorados.")

    # Com --move, remove as pastas de origem que ficaram vazias
    if args.move:
        for source_arg in args.sources:
            source = Path(source_arg)
            for folder in sorted(source.rglob("*"), reverse=True):
                if folder.is_dir() and not any(folder.iterdir()):
                    folder.rmdir()
            if source.is_dir() and not any(source.iterdir()):
                source.rmdir()


if __name__ == "__main__":
    main()
