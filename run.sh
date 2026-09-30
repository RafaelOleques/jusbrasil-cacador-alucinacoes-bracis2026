#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Uso: bash run.sh <caminho_db> <pasta_txt> <arquivo_saida.csv>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB="$1"
TXT="$2"
CSV="$3"
JSON_DIR="${CSV}.json"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

exec python "$ROOT/executar_submissao_ensemble.py" \
  "$TXT" "$DB" "$JSON_DIR" \
  --modelos "$ROOT/modelos_finais" \
  --artefatos "$ROOT/artefatos_finais" \
  --politica agressiva \
  --com-confianca \
  --csv "$CSV"
