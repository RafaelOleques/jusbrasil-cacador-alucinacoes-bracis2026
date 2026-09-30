# -*- coding: utf-8 -*-
"""Classes mínimas necessárias para carregar e usar os artefatos em runtime."""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


class SpanTypeClassifier:
    """Classifica o tipo depois que a fronteira do span já foi encontrada."""

    def __init__(self):
        self.modelo = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                            sublinear_tf=True),
            LogisticRegression(C=2, class_weight="balanced", max_iter=1000,
                               random_state=20260930))

    @staticmethod
    def _x(texto, ini, fim):
        span = texto[ini:fim]
        esquerda = texto[max(0, ini - 45):ini]
        direita = texto[fim:min(len(texto), fim + 45)]
        return f"CTX {esquerda} [SPAN] {span} [/SPAN] {direita}"

    def fit(self, textos, ouro):
        X, y = [], []
        for d in sorted(textos):
            for a in ouro.get(d, []):
                ini, fim = int(a["inicio"]), int(a["fim"])
                if textos[d][ini:fim] != a["trecho"].replace("\\n", "\n"):
                    continue
                X.append(self._x(textos[d], ini, fim))
                y.append(a["tipo"])
        self.modelo.fit(X, y)
        return self

    def predict(self, texto, spans):
        if not spans:
            return []
        tipos = self.modelo.predict(
            [self._x(texto, p["inicio"], p["fim"]) for p in spans])
        return [{**p, "tipo": str(tp)} for p, tp in zip(spans, tipos)]


class SearchabilityClassifier:
    """Estima se o span contém informação suficiente para consultar a base."""

    def __init__(self):
        self.model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                            sublinear_tf=True),
            LogisticRegression(C=2, class_weight="balanced", max_iter=1200,
                               random_state=20260826))

    @staticmethod
    def _x(texto, ini, fim):
        span = texto[ini:fim]
        mascarado = re.sub(r"\d", "#", span)
        esquerda = texto[max(0, ini - 35):ini]
        direita = texto[fim:min(len(texto), fim + 35)]
        return f"{esquerda} [SPAN] {mascarado} [/SPAN] {direita}"

    def fit(self, textos, ouro):
        X, y = [], []
        for d in sorted(textos):
            for g in ouro.get(d, []):
                ini, fim = int(g["inicio"]), int(g["fim"])
                if textos[d][ini:fim] != g["trecho"].replace("\\n", "\n"):
                    continue
                X.append(self._x(textos[d], ini, fim))
                y.append(g["classificacao"] != "incompleta")
        self.model.fit(X, y)
        self.n_train = len(y)
        return self

    def predict(self, texto, spans):
        if not spans:
            return []
        return [bool(x) for x in self.model.predict([
            self._x(texto, p["inicio"], p["fim"]) for p in spans])]
