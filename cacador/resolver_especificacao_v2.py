#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolucao DB-only de citacoes juridicas.

Todos os IDs canonicos e todas as identidades de sumulas/dispositivos sao
reconstruidos do SQLite recebido em runtime. Nao ha IDs copiados da base de
desenvolvimento.
"""
from __future__ import annotations

import collections
import re
import sqlite3
import unicodedata


OCR_OFICIAL = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1",
                             "S": "5", "s": "5"})
D = r"[0-9OolISs]"

RE_CNJ_ISOLADO = re.compile(
    rf"(?<![A-Za-zÀ-ÿ])(?:"
    rf"{D}{{1,7}}\s*-\s*{D}\s*{D}[.\s\-–—]*{D}{{4}}[.\s\-–—]*"
    rf"{D}[.\s\-–—]*{D}{{2}}[.\s\-–—]*{D}{{4}}|{D}{{18,20}})"
    rf"(?![A-Za-zÀ-ÿ])")
RE_AGRUPADO_ISOLADO = re.compile(
    rf"(?<![A-Za-zÀ-ÿ])(?:{D}{{1,3}}(?:[.\s]+{D}{{3}})+|{D}{{4,8}})(?![A-Za-zÀ-ÿ])")
RE_UMA_LETRA_EM_NUMERO = re.compile(
    r"(?<![A-Za-zÀ-ÿ])\d[\d.\s/\-–—]{0,12}[A-Za-z][\d.\s/\-–—]{0,12}\d(?![A-Za-zÀ-ÿ])")
RE_CASO_CABECALHO = re.compile(
    r"(?:reclama[cç][aã]o|recurso(?:\s+especial|\s+extraordin[aá]rio|\s+de\s+revista)?|"
    r"agravo|habeas\s+corpus|mandado\s+de\s+seguran[cç]a|representa[cç][aã]o|processo)"
    r".{0,190}?(?:n[ºo°.]?\s*)?((?:[0-9OolISs][0-9OolISs.\s/\-–—]{3,70}[0-9OolISs]))",
    re.I | re.S)
RE_VISTOS = re.compile(r"vistos\s*,?\s*relatados.{0,500}?autos\s+de(.{0,420})", re.I | re.S)

RE_ARTIGO = re.compile(r"\bart(?:igo)?s?\.?\s*(\d{1,4})(?:\.(\d{3}))?", re.I)
RE_SUMULA = re.compile(
    r"\bS[uú]m(?:ula)?\.?\s*(Vinculante\s*)?(?:n[ºo°.]?\s*)?(\d{1,4})", re.I)
RE_CORTE = re.compile(r"\b(STF|STJ|TST|TSE|STM)\b", re.I)

# Apelidos juridicos estaveis por identidade do diploma, nunca por ID da base.
# Numeros explicitamente citados continuam funcionando para qualquer diploma
# que nao esteja nesta tabela.
APELIDO_POR_DIPLOMA = {
    ("constituicao", "1988"): {"cf"},
    ("lei", "4737"): {"ce"},
    ("decreto-lei", "1001"): {"cpm"},
    ("lei", "8078"): {"cdc"},
    ("decreto-lei", "5452"): {"clt"},
    ("decreto-lei", "3689"): {"cpp"},
    ("lei", "10406"): {"cc"},
    ("lei-complementar", "64"): {"lc64"},
    ("lei", "13105"): {"cpc"},
}

EXPRESSOES_APELIDO = {
    "cf": (r"\bCF(?:/88)?\b", r"constitui[cç][aã]o\s+(?:federal|da\s+rep[uú]blica)"),
    "ce": (r"\bCE\b", r"c[oó]digo\s+eleitoral"),
    "cpm": (r"\bCPM\b", r"c[oó]digo\s+penal\s+militar"),
    "cdc": (r"\bCDC\b", r"c[oó]digo\s+de\s+defesa\s+do\s+consumidor"),
    "clt": (r"\bCLT\b", r"consolida[cç][aã]o\s+das\s+leis\s+do\s+trabalho"),
    "cpp": (r"\bCPP\b", r"c[oó]digo\s+de\s+processo\s+penal"),
    "cc": (r"\bCC\b", r"c[oó]digo\s+civil"),
    "lc64": (r"\bLC\s*64\b",),
    "cpc": (r"\bCPC(?:/15)?\b", r"c[oó]digo\s+de\s+processo\s+civil"),
}


def _sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower()


def _numero(s: str) -> str:
    return "".join(c for c in s if c.isdigit()).lstrip("0") or "0"


def _identidades_diploma(texto: str) -> set[tuple[str, str]]:
    """Extrai identificadores legislativos da superficie ou cabecalho."""
    normal = _sem_acento(texto)
    ids: set[tuple[str, str]] = set()
    padroes = (
        ("lei-complementar", r"lei\s+complementar\s+(?:n(?:[.oº°]|umero)?\s*)?([\d.]+)"),
        ("decreto-lei", r"decreto[\s-]*lei\s+(?:n(?:[.oº°]|umero)?\s*)?([\d.]+)"),
        ("lei", r"(?<!decreto-)\blei\s+(?:n(?:[.oº°]|umero)?\s*)?([\d.]+)"),
    )
    for tipo, padrao in padroes:
        for m in re.finditer(padrao, normal):
            ids.add((tipo, _numero(m.group(1))))
    if re.search(r"constituicao\s+(?:federal|da\s+republica)(?:\s+de\s+1988)?", normal):
        ids.add(("constituicao", "1988"))
    return ids


def _apelidos(texto: str) -> set[str]:
    return {apelido for apelido, padroes in EXPRESSOES_APELIDO.items()
            if any(re.search(p, texto, re.I) for p in padroes)}


def _numero_artigo(texto: str) -> int | None:
    m = RE_ARTIGO.search(texto)
    if not m:
        return None
    return int(m.group(1) + (m.group(2) or ""))


def _corridas(s: str):
    candidatos = []
    for eh_cnj, rex in ((True, RE_CNJ_ISOLADO), (False, RE_AGRUPADO_ISOLADO)):
        for m in rex.finditer(s):
            bruto = m.group()
            chars = [c.translate(OCR_OFICIAL) for c in bruto]
            dig = "".join(c for c in chars if c.isdigit())
            if dig:
                chave = (dig.rjust(20, "0") if eh_cnj and 14 <= len(dig) <= 20
                         else (dig.lstrip("0") or "0"))
                candidatos.append((m.start(), chave, bruto))
    for m in RE_UMA_LETRA_EM_NUMERO.finditer(s):
        bruto = m.group()
        norm = []
        for c in bruto:
            if c.isdigit():
                norm.append(c)
            elif c in "OolISs":
                norm.append(c.translate(OCR_OFICIAL))
            elif c.isalpha():
                norm.append("?")
        chave = "".join(norm).lstrip("0") or "0"
        if chave.count("?") == 1:
            candidatos.append((m.start(), chave, bruto))
    unicos = {(pos, dig): bruto for pos, dig, bruto in candidatos}
    return [(p, d, b) for (p, d), b in sorted(unicos.items())]


def chave_numerica(s: str) -> str:
    candidatos = _corridas(s)
    if not candidatos:
        return ""
    cnj = [x for x in candidatos if 18 <= len(x[1]) <= 20]
    pool = cnj or candidatos
    return max(pool, key=lambda x: (len(x[1]), -x[0]))[1]


def _distancia_ate_1(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 1:
        return 2
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b))
    if len(a) > len(b):
        a, b = b, a
    i = j = erros = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            erros += 1
            j += 1
            if erros > 1:
                return 2
    return erros + (j < len(b))


class IndicePrincipal:
    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.registros = {}
        self.ids_validos: set[int] = set()
        self.por_chave = collections.defaultdict(list)
        self.por_tamanho = collections.defaultdict(set)
        self.feito_de = {}
        self.sumulas: list[dict] = []
        self.dispositivos: list[dict] = []
        self._construir()

    @staticmethod
    def _principal(texto: str, tribunal: str) -> str:
        mv = RE_VISTOS.search(texto[:5000])
        if mv:
            k = chave_numerica(mv.group(1))
            if len(k) >= 5:
                return k
        mc = RE_CASO_CABECALHO.search(texto[:1400])
        if mc:
            k = chave_numerica(mc.group(1))
            if len(k) >= 5:
                return k
        if tribunal == "STJ":
            mn = re.search(r"n[ºo°.]\s*([^A-Za-zÀ-ÿ]{0,3}[0-9OolISs]"
                           r"[0-9OolISs.\s\-–—]{2,20}[0-9OolISs])", texto[:260], re.I)
            if mn:
                k = chave_numerica(mn.group(1))
                if len(k) >= 3:
                    return k
        if tribunal in {"STF", "STJ"}:
            cs = _corridas(texto[:320])
            plausiveis = [x for x in cs if not (
                len(x[1]) == 4 and 1900 <= int(x[1].replace("?", "0")) <= 2100)]
            if plausiveis:
                return max(plausiveis, key=lambda x: (len(x[1]), -x[0]))[1]
        cs = _corridas(texto[:2000])
        cnj = [x for x in cs if 18 <= len(x[1]) <= 20]
        return cnj[0][1] if cnj else ""

    def _construir(self):
        try:
            rows = self.db.execute(
                "select documento_id,id,tribunal,natureza,tipo,texto from documentos")
        except sqlite3.Error as exc:
            raise ValueError(f"schema SQLite incompativel: {exc}") from exc
        for row in rows:
            r = dict(row)
            r["id"] = int(r["id"])
            self.registros[r["documento_id"]] = r
            self.ids_validos.add(r["id"])
            if r["natureza"] == "sumula":
                m = RE_SUMULA.search(r["texto"][:400])
                if m:
                    r["numero_sumula"] = int(m.group(2))
                    r["vinculante"] = bool(m.group(1))
                    corte = r.get("tribunal") or ""
                    if not corte and (mc := RE_CORTE.search(r["texto"][:400])):
                        corte = mc.group(1).upper()
                    r["tribunal_sumula"] = corte.upper()
                    self.sumulas.append(r)
                continue
            if r["natureza"] == "dispositivo":
                numero = _numero_artigo(r["texto"][:400])
                if numero is not None:
                    ids_diploma = _identidades_diploma(r["texto"][:400])
                    apelidos = set().union(*(APELIDO_POR_DIPLOMA.get(x, set())
                                            for x in ids_diploma))
                    r["numero_artigo"] = numero
                    r["ids_diploma"] = ids_diploma
                    r["apelidos_diploma"] = apelidos
                    self.dispositivos.append(r)
                continue
            if r["natureza"] != "acordao":
                continue
            k = self._principal(r["texto"], r.get("tribunal") or "")
            if not k:
                continue
            self.por_chave[k].append(r)
            self.por_tamanho[len(k)].add(k)
            self.feito_de[r["documento_id"]] = (r["tribunal"], k)

    def resolver_sumula(self, trecho: str) -> tuple[int | None, list[int]]:
        m = RE_SUMULA.search(trecho)
        if not m:
            return None, []
        vinculante = bool(m.group(1))
        numero = int(m.group(2))
        corte_m = RE_CORTE.search(trecho)
        corte = corte_m.group(1).upper() if corte_m else ("STF" if vinculante else None)
        achados = [r["id"] for r in self.sumulas
                   if r["numero_sumula"] == numero
                   and r["vinculante"] == vinculante
                   and (corte is None or r["tribunal_sumula"] == corte)]
        return numero, sorted(set(achados))

    def resolver_dispositivo(self, trecho: str) -> tuple[int | None, list[int]]:
        numero = _numero_artigo(trecho)
        if numero is None:
            return None, []
        ids_diploma = _identidades_diploma(trecho)
        apelidos = _apelidos(trecho)
        achados = [r for r in self.dispositivos if r["numero_artigo"] == numero]
        if ids_diploma:
            achados = [r for r in achados if r["ids_diploma"] & ids_diploma]
        if apelidos:
            achados = [r for r in achados if r["apelidos_diploma"] & apelidos]
        return numero, sorted({r["id"] for r in achados})

    def buscar(self, k: str):
        if not k:
            return []
        if k in self.por_chave:
            return self.por_chave[k]
        if len(k) < 6:
            return []
        vizinhas = []
        for n in (len(k) - 1, len(k), len(k) + 1):
            for cand in self.por_tamanho.get(n, ()):
                if _distancia_ate_1(k, cand) <= 1:
                    vizinhas.append(cand)
        if len(set(vizinhas)) != 1:
            return []
        return self.por_chave[vizinhas[0]]

    def agrupar(self, regs):
        grupos = collections.defaultdict(list)
        for r in regs:
            grupos[self.feito_de.get(r["documento_id"],
                                     (r["tribunal"], r["documento_id"]))].append(r)
        return grupos


def qualidade_registro(reg: dict) -> tuple:
    texto = reg["texto"]
    corrompidos = texto.count("\ufffd")
    return (corrompidos / max(1, len(texto)), corrompidos,
            -len(texto), reg["documento_id"])


def representante_canonico(regs):
    return min(regs, key=qualidade_registro)


def classificar_v2(c: dict, indice: IndicePrincipal) -> dict:
    familia = c["familia"]
    if familia in {"formula", "ml_formula"}:
        return {"classificacao": "incompleta", "id_canonico": None, "confianca": .70}
    if familia == "sumula":
        num, ids = indice.resolver_sumula(c["trecho"])
    elif familia == "dispositivo":
        num, ids = indice.resolver_dispositivo(c["trecho"])
    else:
        k = chave_numerica(c["trecho"])
        grupos = indice.agrupar(indice.buscar(k))
        if len(grupos) == 1:
            reg = representante_canonico(next(iter(grupos.values())))
            return {"classificacao": "real", "id_canonico": reg["id"],
                    "confianca": .85}
        return {"classificacao": "inventada" if not grupos else "incompleta",
                "id_canonico": None, "confianca": .65}
    if num is None or len(ids) > 1:
        return {"classificacao": "incompleta", "id_canonico": None, "confianca": .65}
    if not ids:
        return {"classificacao": "inventada", "id_canonico": None, "confianca": .75}
    return {"classificacao": "real", "id_canonico": ids[0], "confianca": .85}


def chave_contextual(texto: str, inicio: int, fim: int) -> str:
    a, b = max(0, inicio - 24), min(len(texto), fim + 72)
    candidatos = []
    for pos, k, bruto in _corridas(texto[a:b]):
        x, y = a + pos, a + pos + len(bruto)
        overlap = max(0, min(fim, y) - max(inicio, x))
        distancia = 0 if overlap else min(abs(x - fim), abs(inicio - y))
        if overlap or distancia <= 4:
            candidatos.append((bool(overlap), len(k), -distancia, -pos, k))
    return max(candidatos)[-1] if candidatos else chave_numerica(texto[inicio:fim])


def classificar_v2_contexto(c: dict, indice: IndicePrincipal, texto: str) -> dict:
    if c["familia"] in {"formula", "ml_formula", "sumula", "dispositivo"}:
        return classificar_v2(c, indice)
    k = chave_contextual(texto, int(c["inicio"]), int(c["fim"]))
    grupos = indice.agrupar(indice.buscar(k))
    if len(grupos) == 1:
        reg = representante_canonico(next(iter(grupos.values())))
        return {"classificacao": "real", "id_canonico": reg["id"], "confianca": .85}
    return {"classificacao": "inventada" if not grupos else "incompleta",
            "id_canonico": None, "confianca": .65}
