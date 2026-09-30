#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ponto de entrada da submissão neural final."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def run(*args):
    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    subprocess.run([sys.executable, *(str(x) for x in args)], check=True, env=env)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloco in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def verificar_modelos(pasta: Path) -> None:
    manifesto = pasta / "manifest.json"
    checksums = pasta / "SHA256SUMS"
    if not manifesto.is_file() or not checksums.is_file():
        raise ValueError(f"artefatos de modelo incompletos em {pasta}")
    dados = json.loads(manifesto.read_text(encoding="utf-8"))
    if len(dados.get("models", [])) != 5:
        raise ValueError("manifesto deve declarar os cinco modelos do ensemble")
    for linha in checksums.read_text(encoding="utf-8").splitlines():
        esperado, relativo = linha.split(maxsplit=1)
        arquivo = pasta / relativo.strip().lstrip("*")
        if not arquivo.is_file():
            raise ValueError(f"peso ausente: {arquivo}")
        observado = sha256(arquivo)
        if observado != esperado.lower():
            raise ValueError(f"hash divergente: {arquivo}")


def verificar_artefatos(pasta: Path) -> None:
    manifesto = pasta / "manifest.json"
    if not manifesto.is_file():
        raise ValueError(f"manifesto de artefatos ausente: {manifesto}")
    dados = json.loads(manifesto.read_text(encoding="utf-8"))
    for relativo, esperado in dados.get("files", {}).items():
        arquivo = pasta / relativo
        if not arquivo.is_file() or sha256(arquivo) != esperado.lower():
            raise ValueError(f"artefato ausente ou corrompido: {arquivo}")


def verificar_entrada(txt: Path, db: Path, saida: Path, csv: Path | None) -> None:
    if not txt.is_dir():
        raise ValueError(f"pasta de textos nao encontrada: {txt}")
    textos = sorted(txt.glob("*.txt"))
    if not textos:
        raise ValueError(f"nenhum .txt encontrado em {txt}")
    if len({p.stem for p in textos}) != len(textos):
        raise ValueError("nomes de documentos colidem depois da remocao de .txt")
    if not db.is_file():
        raise ValueError(f"base nao encontrada: {db}")
    con = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    try:
        colunas = {r[1] for r in con.execute("pragma table_info(documentos)")}
    finally:
        con.close()
    exigidas = {"documento_id", "id", "tribunal", "natureza", "tipo", "texto"}
    if not exigidas <= colunas:
        raise ValueError(f"schema SQLite incompativel; faltam {sorted(exigidas-colunas)}")
    if saida.exists() and any(saida.iterdir()):
        raise ValueError(f"pasta de JSON deve estar vazia: {saida}")
    if csv is not None and csv.exists():
        raise ValueError(f"arquivo de saida ja existe: {csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("txt", type=Path, nargs="?")
    ap.add_argument("db", type=Path, nargs="?")
    ap.add_argument("saida", type=Path, nargs="?")
    ap.add_argument("--input", type=Path,
                    help="diretorio do kit: txt/ e desafio1_bracis.db")
    ap.add_argument("--output", type=Path,
                    help="diretorio onde JSONs e submission.csv serao escritos")
    ap.add_argument("--modelos", type=Path)
    ap.add_argument("--artefatos", type=Path)
    ap.add_argument("--politica", choices=("conservadora", "agressiva"),
                    default="agressiva")
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--com-confianca", action="store_true")
    ap.add_argument("--teto-confianca", type=float, default=None)
    ap.add_argument("--expandir-direita", type=int, default=0)
    ap.add_argument("--consenso-com-digito", type=int, default=3)
    ap.add_argument("--consenso-sem-digito", type=int, default=4)
    ap.add_argument("--permitir-cpu", action="store_true",
                    help="fallback lento, somente para diagnostico")
    a = ap.parse_args()
    if a.input is not None or a.output is not None:
        if a.input is None or a.output is None or any(x is not None for x in (a.txt, a.db, a.saida)):
            ap.error("use --input e --output juntos, sem os tres posicionais")
        a.txt = a.input / "txt" if (a.input / "txt").is_dir() else a.input
        a.db = a.input / "desafio1_bracis.db"
        a.saida = a.output / "json"
        if a.csv is None:
            a.csv = a.output / "submission.csv"
    elif any(x is None for x in (a.txt, a.db, a.saida)):
        ap.error("informe --input/--output ou TXT DB SAIDA")
    raiz = Path(__file__).resolve().parent
    a.modelos = a.modelos or raiz / "modelos_finais"
    a.artefatos = a.artefatos or raiz / "artefatos_finais"
    try:
        verificar_entrada(a.txt, a.db, a.saida, a.csv)
        verificar_modelos(a.modelos)
        verificar_artefatos(a.artefatos)
    except (OSError, ValueError, sqlite3.Error, json.JSONDecodeError) as exc:
        ap.error(str(exc))
    a.saida.mkdir(parents=True, exist_ok=True)
    if a.csv:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [raiz / "cacador" / "prever_ensemble_final.py", a.txt, a.db,
           a.modelos, a.artefatos, a.saida, "--politica", a.politica]
    if a.com_confianca:
        cmd.append("--com-confianca")
    if a.teto_confianca is not None:
        cmd.extend(("--teto-confianca", str(a.teto_confianca)))
    if a.expandir_direita:
        cmd.extend(("--expandir-direita", str(a.expandir_direita)))
    cmd.extend(("--consenso-com-digito", str(a.consenso_com_digito),
                "--consenso-sem-digito", str(a.consenso_sem_digito)))
    if a.permitir_cpu:
        cmd.append("--permitir-cpu")
    run(*cmd)
    run(raiz / "validar_saida.py", a.txt, a.saida, a.db)
    if a.csv:
        run(raiz / "json_to_submission.py", a.saida, a.csv)


if __name__ == "__main__":
    main()
