#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Valida os JSONs de predicao contra os textos e a base, sem usar gold."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

CLASSES = {"real", "inventada", "incompleta"}
TIPOS = {"lei", "jurisprudencia"}


def validar(txt_dir: Path, preds_dir: Path, db_path: Path) -> list[str]:
    erros: list[str] = []
    textos = {p.stem: p for p in txt_dir.glob("*.txt")}
    preds = {p.stem: p for p in preds_dir.glob("*.json")}
    if set(textos) != set(preds):
        for nome in sorted(set(textos) - set(preds)):
            erros.append(f"{nome}: JSON ausente")
        for nome in sorted(set(preds) - set(textos)):
            erros.append(f"{nome}: JSON sem TXT correspondente")

    with sqlite3.connect(db_path) as con:
        ids_validos = {str(r[0]) for r in con.execute("SELECT id FROM documentos")}

    for nome in sorted(set(textos) & set(preds)):
        texto = textos[nome].read_text(encoding="utf-8")
        try:
            doc = json.loads(preds[nome].read_text(encoding="utf-8"))
        except Exception as exc:
            erros.append(f"{nome}: JSON invalido ({exc})")
            continue
        if doc.get("documento_id") != nome:
            erros.append(f"{nome}: documento_id divergente")
        citacoes = doc.get("citacoes")
        if not isinstance(citacoes, list):
            erros.append(f"{nome}: citacoes deve ser uma lista")
            continue
        ids_citacao: set[str] = set()
        spans: list[tuple[int, int]] = []
        for pos, c in enumerate(citacoes, 1):
            rotulo = f"{nome}/citacao[{pos}]"
            cid = c.get("citacao_id")
            if not isinstance(cid, str) or not cid or cid in ids_citacao:
                erros.append(f"{rotulo}: citacao_id ausente ou duplicado")
            ids_citacao.add(cid)
            inicio, fim = c.get("inicio"), c.get("fim")
            if (not isinstance(inicio, int) or isinstance(inicio, bool)
                    or not isinstance(fim, int) or isinstance(fim, bool)
                    or not 0 <= inicio < fim <= len(texto)):
                erros.append(f"{rotulo}: span invalido")
                continue
            if c.get("trecho") != texto[inicio:fim]:
                erros.append(f"{rotulo}: trecho nao corresponde ao span")
            if c.get("tipo") not in TIPOS:
                erros.append(f"{rotulo}: tipo invalido")
            classe = c.get("classificacao")
            if classe not in CLASSES:
                erros.append(f"{rotulo}: classificacao invalida")
            resolucao = c.get("resolucao")
            if classe == "real":
                rid = str((resolucao or {}).get("id_canonico", ""))
                if rid not in ids_validos:
                    erros.append(f"{rotulo}: real sem id_canonico valido na base")
            elif resolucao is not None:
                erros.append(f"{rotulo}: resolucao deve ser null para classe nao-real")
            if "confianca" in c:
                conf = c["confianca"]
                if (not isinstance(conf, (int, float)) or isinstance(conf, bool)
                        or not 0 <= conf <= 1):
                    erros.append(f"{rotulo}: confianca fora de [0,1]")
            spans.append((inicio, fim))
        if spans != sorted(spans):
            erros.append(f"{nome}: citacoes fora da ordem textual")
    return erros


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("txt", type=Path)
    ap.add_argument("preds", type=Path)
    ap.add_argument("db", type=Path)
    args = ap.parse_args()
    erros = validar(args.txt, args.preds, args.db)
    if erros:
        print("SAIDA INVALIDA:")
        print("\n".join(f"- {e}" for e in erros))
        raise SystemExit(1)
    print(f"SAIDA VALIDA: {len(list(args.preds.glob('*.json')))} documentos")


if __name__ == "__main__":
    main()
