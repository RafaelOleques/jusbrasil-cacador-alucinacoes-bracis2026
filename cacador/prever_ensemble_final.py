#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inferência offline do ensemble final refitado, sem acesso ao gold."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if (RAIZ / ".deps").exists():
    sys.path.insert(0, str(RAIZ / ".deps"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

import runtime_inferencia as rt
from filtros_estruturais import dispositivo_sem_localizador_numerico
from runtime_inferencia import (consenso_condicional_digitos, familia,
                                predizer_skeleton, sem_acento, unir)
from resolver_especificacao_v2 import (IndicePrincipal, chave_numerica,
                                        classificar_v2, classificar_v2_contexto)

RE_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def configurar_determinismo(seed=20260930):
    """Fixa as fontes de aleatoriedade usadas pelo runtime."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def formulas_exatas(texto, itens):
    alvo = sem_acento(texto).lower()
    saida = []
    for item in itens:
        padrao = r"\s+".join(map(re.escape, item["text"].split()))
        for m in re.finditer(padrao, alvo):
            saida.append({"inicio": m.start(), "fim": m.end(),
                          "trecho": texto[m.start():m.end()],
                          "tipo": item["type"], "familia": "ml_formula",
                          "confianca": .99})
    return sorted(saida, key=lambda x: x["inicio"])


def resolver(texto, spans, busca, indice):
    xs = [{**p, "familia": familia(p)} for p in spans]
    flags = busca.predict(texto, xs)
    saida = []
    for p, buscavel_ml in zip(xs, flags):
        confianca_extracao = float(p.get("confianca", .65))
        sem_digito_util = (p["familia"] not in {"sumula", "dispositivo"}
                           and re.search(r"\d", RE_YEAR.sub(" ", p["trecho"])) is None)
        buscavel = (p["familia"] in {"sumula", "dispositivo"}
                    or len(chave_numerica(p["trecho"])) >= 5
                    or buscavel_ml)
        if sem_digito_util or not buscavel:
            r = {"classificacao": "incompleta", "id_canonico": None,
                 "confianca": .70}
        else:
            r = classificar_v2_contexto(p, indice, texto)
        # A confiança OOF promovida em E31 é da emissão end-to-end aproximada
        # pela evidência neural. A constante do resolvedor é pior calibrada e
        # não deve sobrescrevê-la.
        saida.append({**p, **r, "confianca": confianca_extracao})
    return saida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("txt", type=Path)
    ap.add_argument("db", type=Path)
    ap.add_argument("modelos", type=Path)
    ap.add_argument("artefatos", type=Path)
    ap.add_argument("saida", type=Path)
    ap.add_argument("--politica", choices=("conservadora", "agressiva"),
                    default="agressiva")
    ap.add_argument("--com-confianca", action="store_true")
    ap.add_argument("--teto-confianca", type=float, default=None,
                    help="teto opcional; omitir preserva a média bruta E31")
    ap.add_argument("--expandir-direita", type=int, default=0,
                    help="margem pós-resolução no fim do span; 0 desativa")
    ap.add_argument("--consenso-com-digito", type=int, default=3,
                    help="votos neurais mínimos para spans com dígitos (E64)")
    ap.add_argument("--consenso-sem-digito", type=int, default=4,
                    help="votos neurais mínimos para spans sem dígitos (E64)")
    ap.add_argument("--permitir-cpu", action="store_true",
                    help="permite fallback lento sem CUDA (somente diagnostico)")
    a = ap.parse_args()
    configurar_determinismo()
    arquivos = sorted(a.txt.glob("*.txt"))
    if not arquivos:
        ap.error("nenhum .txt encontrado")
    textos = {p.stem: p.read_text(encoding="utf-8") for p in arquivos}
    manifest = json.loads((a.modelos / "manifest.json").read_text(encoding="utf-8"))
    estrutural = json.loads((a.artefatos / "estrutural.json").read_text(encoding="utf-8"))
    tipo = joblib.load(a.artefatos / "type_classifier.joblib")
    busca = joblib.load(a.artefatos / "searchability_classifier.joblib")
    tokenizer = AutoTokenizer.from_pretrained(
        a.modelos / "tokenizer", use_fast=True, add_prefix_space=True,
        local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not a.permitir_cpu:
        ap.error("GPU CUDA nao encontrada; use --permitir-cpu apenas para diagnostico")
    fontes = []
    for cfg in manifest["models"]:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        modelo = AutoModelForTokenClassification.from_pretrained(
            a.modelos / cfg["path"], torch_dtype=dtype,
            local_files_only=True).to(device)
        pp = {}
        for d, texto in textos.items():
            toks, probs = rt.probabilidades_palavra(
                modelo, tokenizer, d, texto, device)
            pp[d] = rt.spans_de_prob(texto, toks, probs, cfg["threshold"])
            if cfg["boundary"]:
                pp[d] = tipo.predict(texto, pp[d])
        fontes.append(pp)
        del modelo
        if device.type == "cuda":
            torch.cuda.empty_cache()
    neural = {
        d: consenso_condicional_digitos(
            [p[d] for p in fontes], textos[d],
            votos_com_digito=a.consenso_com_digito,
            votos_sem_digito=a.consenso_sem_digito)
        for d in sorted(textos)
    }
    for d, ps in neural.items():
        for p in ps:
            confs = []
            for fonte in fontes:
                candidatos = [q for q in fonte[d]
                              if rt.iou((p["inicio"], p["fim"]),
                                        (q["inicio"], q["fim"])) >= .5]
                if candidatos:
                    q = max(candidatos, key=lambda q: rt.iou(
                        (p["inicio"], p["fim"]), (q["inicio"], q["fim"])))
                    confs.append(float(q.get("confianca", .5)))
            if not confs:
                raise AssertionError("span do consenso sem modelo apoiador")
            p["confianca"] = sum(confs) / len(confs)
    templates = {(tuple(x["prefix"]), int(x["name_tokens"])): int(x["support"])
                 for x in estrutural["narrative_skeleton"]}
    brutos = {}
    for d, texto in textos.items():
        formulas = formulas_exatas(texto, estrutural["formula_exact"])
        p = unir(neural[d], formulas)
        if a.politica == "agressiva":
            p = unir(p, predizer_skeleton(
                texto, templates, estrutural["narrative_skeleton_min_support"]))
        brutos[d] = [x for x in p
                     if not dispositivo_sem_localizador_numerico(x, texto)]
    indice = IndicePrincipal(str(a.db))
    resolvidos = {d: resolver(textos[d], brutos[d], busca, indice)
                  for d in sorted(textos)}
    indice.db.close()
    if a.expandir_direita:
        if a.expandir_direita < 0:
            raise ValueError("--expandir-direita deve ser não negativo")
        for d, ps in resolvidos.items():
            for p in ps:
                p["fim"] = min(len(textos[d]), int(p["fim"]) + a.expandir_direita)
                p["trecho"] = textos[d][int(p["inicio"]):int(p["fim"])]
    a.saida.mkdir(parents=True, exist_ok=True)
    for d, ps in resolvidos.items():
        citacoes = []
        for i, p in enumerate(sorted(ps, key=lambda x: x["inicio"]), 1):
            c = {"citacao_id": f"c{i}", "inicio": int(p["inicio"]),
                 "fim": int(p["fim"]), "trecho": p["trecho"],
                 "tipo": p["tipo"], "classificacao": p["classificacao"],
                 "resolucao": ({"id_canonico": int(p["id_canonico"])}
                                if p.get("id_canonico") else None)}
            if a.com_confianca:
                conf = float(p.get("confianca", .65))
                if a.teto_confianca is not None:
                    conf = min(conf, a.teto_confianca)
                c["confianca"] = round(conf, 4)
            citacoes.append(c)
        (a.saida / f"{d}.json").write_text(
            json.dumps({"documento_id": d, "citacoes": citacoes},
                       ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{a.saida}: {len(resolvidos)} documentos; politica={a.politica}")


if __name__ == "__main__":
    main()
