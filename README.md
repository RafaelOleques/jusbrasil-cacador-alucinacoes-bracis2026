# Caça-Alucinações — entrega BRACIS 2026

Pipeline offline para localizar citações em documentos jurídicos, classificá-las como `real`, `inventada` ou `incompleta` e resolver, quando possível, um único `id_canonico` presente no SQLite fornecido pela organização.

## Comando único

O contrato de entrega é:

```bash
bash run.sh <caminho_db> <pasta_txt> <arquivo_saida.csv>
```

Exemplo:

```bash
bash run.sh /dados/desafio.db /dados/txt /saida/submission.csv
```

O comando cria os JSONs intermediários em `<arquivo_saida.csv>.json`, valida cada documento contra os textos e o SQLite e só então grava o CSV. A pasta de JSON deve estar vazia antes de cada execução e o CSV de destino não pode existir, evitando misturar resultados de rodadas diferentes.

## Docker

Os pesos e os artefatos auxiliares estão dentro da imagem e os cinco arquivos `.safetensors` são rastreados por Git LFS. Depois de clonar o repositório e baixar os objetos LFS:

```bash
docker build -t cacador-bracis:2026 .
docker run --rm --gpus all --network none \
  -v /dados:/dados:ro -v /saida:/saida \
  cacador-bracis:2026 /dados/desafio.db /dados/txt /saida/submission.csv
```

O runtime exige CUDA por padrão, usa no máximo uma GPU e foi dimensionado para até 24 GB de VRAM. Para diagnóstico local sem GPU, o executor Python aceita `--permitir-cpu`, mas esse modo não é o modo oficial de avaliação.

## Ambiente e reprodutibilidade

- imagem base imutável: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee`;
- dependências fixadas em `requirements-submission.txt`;
- pesos e tokenizer locais, sem download em runtime;
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, seeds fixas e algoritmos determinísticos;
- `modelos_finais/SHA256SUMS` e `artefatos_finais/manifest.json` verificados antes da inferência;
- o índice canônico, inclusive súmulas e dispositivos, é reconstruído a partir do SQLite recebido. Nenhum ID da base de desenvolvimento é usado.

## Estrutura

- `run.sh`: ponto de entrada solicitado pela organização;
- `executar_submissao_ensemble.py`: pré-validação, inferência, validação e CSV;
- `cacador/`: módulos mínimos de runtime, sem treino, gold ou avaliação;
- `modelos_finais/`: cinco modelos finais e tokenizer;
- `artefatos_finais/`: classificadores e regras estruturais finais;
- `tests/test_runtime_db.py`: teste de portabilidade com IDs de uma base nova.

## Verificações locais

```bash
python -m unittest discover -s tests -v
python -m py_compile executar_submissao_ensemble.py json_to_submission.py \
  validar_saida.py cacador/*.py
```

O teste de runtime confirma que IDs diferentes dos usados no desenvolvimento são lidos da base recebida.

Em 01/10/2026 o repositório público foi clonado do zero e executado via `bash run.sh` em GPU (NVIDIA RTX 4000 Ada, CUDA), com a base `desafio1_bracis.db` distribuída pela organização. Resultado: 26 JSONs válidos, CSV com 26 documentos, 10 segundos de inferência, SHA-256 do CSV `6afa4f857fccfd0ccd212ede1c761696a763b3169192f970900976c672ba1085`. Essa saída, avaliada localmente com `kaggle_metric.py` contra `goldenset.csv`, obtém score 1,0780 (nível 1: 1,0559; nível 2: 1,0891) — número de desenvolvimento, não o da avaliação oficial, que roda sobre documentos e base inéditos.

## Saída

Cada JSON contém `documento_id`, spans com offsets literais, `trecho`, `tipo`, `classificacao`, `resolucao` e, por padrão, `confianca`. O CSV tem exatamente as colunas `documento_id,citacoes` no formato usado pela submissão.
