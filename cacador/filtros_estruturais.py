#!/usr/bin/env python3
"""Filtros determinísticos derivados do contrato do desafio."""
from __future__ import annotations

import re

RE_LOCALIZADOR_LEGAL = re.compile(
    r"(?i)(?:\b(?:art(?:igo)?s?\.?|lei|decreto|resolu[cç][aã]o)\s*"
    r"(?:n(?:[.º°o]|[úu]mero)?\s*)?(?:\d|[IVXLCDM]+\b)|§\s*(?:\d|[IVXLCDM]+\b))"
)


def dispositivo_sem_localizador_numerico(p, texto=None, janela_esquerda=96):
    """Detecta menção genérica a diploma, sem apontar dispositivo específico.

    A base canônica indexa dispositivos (artigos etc.), não o diploma inteiro.
    Portanto, uma superfície como ``do Código de Processo Civil`` não formula
    uma consulta unívoca. A regra é estrutural: não usa frase, documento,
    offset ou identificador observado no gabarito.
    """
    if p.get("tipo") != "lei" or re.search(r"\d", p["trecho"]):
        return False
    if texto is None or "inicio" not in p:
        return True
    inicio = int(p["inicio"])
    esquerda = texto[max(0, inicio - janela_esquerda):inicio]
    # Não deixa um número de outra oração justificar a menção atual.
    esquerda = re.split(r"[;:]|\n\s*\n", esquerda)[-1]
    return RE_LOCALIZADOR_LEGAL.search(esquerda) is None
