from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cacador"))

from resolver_especificacao_v2 import IndicePrincipal, classificar_v2


class RuntimeDbTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "nova_base.db"
        with sqlite3.connect(self.db) as con:
            con.execute("""create table documentos (
                documento_id text primary key, id integer not null,
                tribunal text, ano integer, relator text,
                natureza text not null, tipo text not null,
                texto text not null, texto_len integer not null)""")
            rows = [
                ("sum_83", 990083, "STJ", None, None, "sumula", "jurisprudencia",
                 "Súmula n. 83 do STJ Texto", 26),
                ("sv_10", 990010, "STF", None, None, "sumula", "jurisprudencia",
                 "Súmula Vinculante n. 10 do STF Texto", 38),
                ("cpc_373", 880373, None, None, None, "dispositivo", "lei",
                 "Artigo 373 da Lei nº 13.105, de 16 de março de 2015 Art. 373.", 68),
                ("cf_5", 880005, None, None, None, "dispositivo", "lei",
                 "Artigo 5º da Constituição Federal de 1988 Art. 5º.", 52),
                ("cdc_14", 880014, None, None, None, "dispositivo", "lei",
                 "Artigo 14 da Lei nº 8.078, de 11 de setembro de 1990 Art. 14.", 65),
                ("outra_14", 770014, None, None, None, "dispositivo", "lei",
                 "Artigo 14 da Lei nº 99.999, de 1 de janeiro de 2000 Art. 14.", 63),
            ]
            con.executemany("insert into documentos values (?,?,?,?,?,?,?,?,?)", rows)
        con.close()
        self.indice = IndicePrincipal(str(self.db))

    def tearDown(self):
        self.indice.db.close()
        self.tmp.cleanup()

    def resolver(self, familia, trecho):
        return classificar_v2({"familia": familia, "trecho": trecho}, self.indice)

    def test_ids_sao_lidos_da_nova_base(self):
        casos = [
            ("sumula", "Súmula 83 do STJ", 990083),
            ("sumula", "Súmula Vinculante 10", 990010),
            ("dispositivo", "art. 373, I, do CPC", 880373),
            ("dispositivo", "artigo 5º da Constituição da República", 880005),
            ("dispositivo", "art. 14 do Código de Defesa do Consumidor", 880014),
        ]
        for familia, trecho, esperado in casos:
            with self.subTest(trecho=trecho):
                observado = self.resolver(familia, trecho)
                self.assertEqual(observado["classificacao"], "real")
                self.assertEqual(observado["id_canonico"], esperado)

    def test_inexistente_e_inventada(self):
        observado = self.resolver("sumula", "Súmula 999 do STJ")
        self.assertEqual(observado["classificacao"], "inventada")
        self.assertIsNone(observado["id_canonico"])

    def test_dispositivo_ambiguo_sem_diploma_e_incompleto(self):
        observado = self.resolver("dispositivo", "art. 14")
        self.assertEqual(observado["classificacao"], "incompleta")
        self.assertIsNone(observado["id_canonico"])


if __name__ == "__main__":
    unittest.main()
