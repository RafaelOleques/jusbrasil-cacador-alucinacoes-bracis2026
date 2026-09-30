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

Em 30/09/2026, o pipeline atual foi executado de ponta a ponta sobre os 26 documentos da amostra, com os cinco pesos E67, os três artefatos finais, a imagem base fixada por digest e exatamente as versões de `requirements-submission.txt`. A execução de verificação usou `--permitir-cpu` porque a cota de GPU do cluster estava ocupada; o modo oficial continua exigindo CUDA por padrão. Resultado: 26 JSONs válidos, CSV com 26 documentos, 37 segundos de inferência e SHA-256 do CSV `cbf970b2eceab1698f00f96067648044b24fbdf3019ebd0b840966e45900a68f`.

## Saída

Cada JSON contém `documento_id`, spans com offsets literais, `trecho`, `tipo`, `classificacao`, `resolucao` e, por padrão, `confianca`. O CSV tem exatamente as colunas `documento_id,citacoes` no formato usado pela submissão.
