FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-submission.txt ./
# PyTorch/CUDA ja fazem parte da imagem base.
RUN grep -v '^torch==' requirements-submission.txt > /tmp/requirements-container.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements-container.txt

COPY executar_submissao_ensemble.py json_to_submission.py validar_saida.py run.sh ./
COPY cacador/prever_ensemble_final.py cacador/runtime_inferencia.py \
     cacador/resolver_especificacao_v2.py cacador/filtros_estruturais.py \
     cacador/runtime_classifiers.py ./cacador/
COPY modelos_finais/ ./modelos_finais/
COPY artefatos_finais/ ./artefatos_finais/

RUN python -m py_compile executar_submissao_ensemble.py json_to_submission.py \
        validar_saida.py cacador/*.py \
    && chmod +x run.sh

ENTRYPOINT ["bash", "/app/run.sh"]
