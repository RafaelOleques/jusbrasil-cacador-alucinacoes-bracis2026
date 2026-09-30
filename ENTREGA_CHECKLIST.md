# Checklist antes do envio

## Concluído nesta versão

- [x] Cinco pesos E67 e três artefatos finais presentes e conferidos por SHA-256.
- [x] Objetos grandes rastreados por Git LFS e `git lfs fsck` executado.
- [x] Testes de portabilidade do SQLite: 3/3 passando com IDs diferentes da base de desenvolvimento.
- [x] Compilação dos módulos Python concluída.
- [x] Smoke de ponta a ponta sobre os 26 documentos, no ambiente declarado: 26 JSONs e CSV válidos.
- [x] Runtime sem caminhos absolutos, chamadas de rede ou IDs canônicos fixos da base de desenvolvimento.
- [x] `run.sh` e `Dockerfile` preservados com final de linha LF em clones Windows.
- [x] Repositório Git limpo de entrega gerado a partir do commit final.

## Antes de enviar o e-mail

1. Criar o repositório remoto e adicionar sua URL como `origin`.
2. Executar `git push -u origin master` e `git lfs push --all origin master`.
3. Conceder leitura aos cinco users indicados pela organização, se o repositório for privado.
4. Confirmar que um clone limpo completa `git lfs pull` e, em uma máquina com Docker/GPU, `docker build -t cacador-bracis:2026 .`.
5. Enviar nome da equipe, integrantes, URL do repositório e o hash de `git rev-parse HEAD`.

O smoke da amostra pública não é a nota final; a organização executará o mesmo comando sobre a nova base e os documentos cegos.
