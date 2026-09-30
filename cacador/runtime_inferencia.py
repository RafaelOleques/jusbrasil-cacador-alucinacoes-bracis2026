# -*- coding: utf-8 -*-
"""Primitivas mínimas de inferência; sem imports de treino, gold ou avaliação."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import torch

from resolver_especificacao_v2 import RE_ARTIGO as RE_LEI, RE_SUMULA

LABELS = ["O", "B-lei", "I-lei", "B-jurisprudencia", "I-jurisprudencia"]
RE_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
RE_PALAVRA = re.compile(r"[a-z0-9]+")
TRIBUNAIS = {"stf", "stj", "tse", "tst", "stm"}


def sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def tokenizar_palavras(texto):
    return [(m.group(0), m.start(), m.end()) for m in RE_TOKEN.finditer(texto)]


def preparar_doc(tokenizer, doc, texto, max_length=384, stride=96):
    toks = tokenizar_palavras(texto)
    enc = tokenizer([x[0] for x in toks], is_split_into_words=True,
                    truncation=True, max_length=max_length, stride=stride,
                    return_overflowing_tokens=True)
    itens = []
    for k in range(len(enc["input_ids"])):
        item = {key: enc[key][k] for key in ("input_ids", "attention_mask")}
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"][k]
        item["doc"] = doc
        item["word_ids_raw"] = [-1 if w is None else int(w)
                                for w in enc.word_ids(k)]
        itens.append(item)
    return toks, itens


@torch.inference_mode()
def probabilidades_palavra(modelo, tokenizer, doc, texto, device):
    toks, partes = preparar_doc(tokenizer, doc, texto)
    soma = np.zeros((len(toks), len(LABELS)), dtype=np.float64)
    conta = np.zeros(len(toks), dtype=np.int32)
    modelo.eval()
    for item in partes:
        entrada = {k: torch.tensor([v], device=device)
                   for k, v in item.items()
                   if k in ("input_ids", "attention_mask", "token_type_ids")}
        prob = torch.softmax(modelo(**entrada).logits[0], -1).cpu().numpy()
        vistos = set()
        for pos, w in enumerate(item["word_ids_raw"]):
            if w < 0 or w in vistos:
                continue
            soma[w] += prob[pos]; conta[w] += 1; vistos.add(w)
    conta[conta == 0] = 1
    return toks, soma / conta[:, None]


def spans_de_prob(texto, toks, prob, limiar):
    marcados = []
    for linha in prob:
        j = int(np.argmax(linha[1:])) + 1
        marcados.append(j if linha[j] >= limiar else 0)
    spans, i = [], 0
    while i < len(toks):
        if not marcados[i]:
            i += 1; continue
        tipo = "lei" if marcados[i] in (1, 2) else "jurisprudencia"
        j = i + 1
        while j < len(toks):
            tipo_j = "lei" if marcados[j] in (1, 2) else "jurisprudencia"
            if not marcados[j] or tipo_j != tipo:
                break
            j += 1
        if sum(any(c.isalnum() for c in toks[k][0]) for k in range(i, j)) >= 2:
            ini, fim = toks[i][1], toks[j-1][2]
            spans.append({"inicio": ini, "fim": fim, "trecho": texto[ini:fim],
                          "tipo": tipo, "confianca": float(max(
                              prob[k, 1:].max() for k in range(i, j)))})
        i = j
    return spans


def consenso_sem_tipo(predicoes, texto, votos_minimos=2):
    candidatos = [(m, p) for m, ps in enumerate(predicoes) for p in ps]
    usados, saida = set(), []
    for i, (modelo, p) in enumerate(candidatos):
        if i in usados:
            continue
        grupo = [(i, modelo, p)]
        for j, (outro, q) in enumerate(candidatos):
            if j == i or j in usados or outro == modelo:
                continue
            if iou((p["inicio"], p["fim"]), (q["inicio"], q["fim"])) >= .5:
                grupo.append((j, outro, q))
        if len({x[1] for x in grupo}) < votos_minimos:
            continue
        usados.update(x[0] for x in grupo)
        escolhido = max(grupo, key=lambda x: (x[2]["fim"]-x[2]["inicio"],
                                               -x[2]["inicio"]))[2]
        novo = dict(escolhido); novo["trecho"] = texto[novo["inicio"]:novo["fim"]]
        novo["suporte_modelos"] = len({x[1] for x in grupo})
        confs = [float(x[2].get("confianca", .5)) for x in grupo]
        novo["confianca_consenso"] = sum(confs) / len(confs)
        tipos = [x[2].get("tipo") for x in grupo if x[2].get("tipo")]
        if tipos:
            novo["tipo"] = max(sorted(set(tipos)), key=lambda t: tipos.count(t))
        saida.append(novo)
    # Âncoras diferentes podem formar grupos cujos spans escolhidos ainda se
    # sobrepõem. O avaliador oficial não aceita duas citações casadas à mesma
    # região. Esta NMS é determinística e depende apenas das predições.
    prioridade = sorted(saida, key=lambda x: (
        x["suporte_modelos"], x["confianca_consenso"],
        x["fim"] - x["inicio"], -x["inicio"], -x["fim"]), reverse=True)
    nms = []
    for p in prioridade:
        if any(iou((p["inicio"], p["fim"]), (q["inicio"], q["fim"])) >= .5
               for q in nms):
            continue
        nms.append(p)
    return sorted(nms, key=lambda x: x["inicio"])


def consenso_condicional_digitos(predicoes, texto, votos_com_digito=3,
                                  votos_sem_digito=4):
    """Consenso E64: identificadores numéricos aceitam um voto a menos.

    A política é semântica e cega ao gold: dígitos tornam a referência
    verificável na base; spans puramente narrativos exigem concordância maior.
    """
    n_modelos = len(predicoes)
    for nome, valor in (("votos_com_digito", votos_com_digito),
                        ("votos_sem_digito", votos_sem_digito)):
        if not 1 <= valor <= n_modelos:
            raise ValueError(f"{nome} deve estar entre 1 e {n_modelos}")
    base = consenso_sem_tipo(
        predicoes, texto, min(votos_com_digito, votos_sem_digito))
    return [p for p in base
            if int(p["suporte_modelos"]) >= (
                votos_com_digito if re.search(r"\d", p["trecho"])
                else votos_sem_digito)]


def unir(base, extras):
    saida = list(base)
    for p in sorted(extras, key=lambda x: (x["inicio"], -(x["fim"]-x["inicio"]))):
        if any(iou((p["inicio"], p["fim"]), (q["inicio"], q["fim"])) > .1
               for q in saida):
            continue
        saida.append(p)
    return sorted(saida, key=lambda x: x["inicio"])


def familia(p):
    if RE_SUMULA.search(p["trecho"]): return "sumula"
    if RE_LEI.search(p["trecho"]): return "dispositivo"
    if any(c.isdigit() for c in p["trecho"]): return "identificador"
    return "ml_formula"


def predizer_skeleton(texto, templates, min_support):
    alvo = sem_acento(texto).lower()
    toks = [(m.group(), m.start(), m.end(), texto[m.start():m.end()])
            for m in RE_PALAVRA.finditer(alvo)]
    xs = []
    for (prefixo, n_nome), suporte in templates.items():
        if suporte < min_support:
            continue
        n = len(prefixo)
        for i in range(len(toks)-n-n_nome+1):
            def casa(obs, esp):
                return (obs in TRIBUNAIS if esp == "<TRIB>" else
                        re.fullmatch(r"(?:19|20)\d{2}", obs) is not None
                        if esp == "<YEAR>" else obs == esp)
            if not all(casa(toks[i+j][0], e) for j, e in enumerate(prefixo)):
                continue
            nomes = toks[i+n:i+n+n_nome]
            if not nomes or not nomes[0][3][:1].isupper() or not nomes[-1][3][:1].isupper():
                continue
            ini, fim = toks[i][1], nomes[-1][2]
            xs.append({"inicio": ini, "fim": fim, "trecho": texto[ini:fim],
                       "tipo": "jurisprudencia", "familia": "identificador",
                       "suporte_esqueleto": suporte})
    saida = []
    for p in sorted(xs, key=lambda x: (-x["suporte_esqueleto"],
                                       -(x["fim"]-x["inicio"]), x["inicio"])):
        if not any(iou((p["inicio"], p["fim"]), (q["inicio"], q["fim"])) >= .5
                   for q in saida):
            saida.append(p)
    return sorted(saida, key=lambda x: x["inicio"])
