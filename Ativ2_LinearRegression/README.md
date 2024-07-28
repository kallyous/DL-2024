# Atividade 2 - Regressão Linear

Lucas Carvalho Flores  
Ufal 2024


## Requisitos

* Python 3
* Pipenv


## Instalação

No diretório raiz do projeto, executar:
```bash
pipenv install
```

## Execução

Ainda na raiz do prjeto, rodar o script `main.py` com:
```bash
pipenv shell
python main.py
```


## Info Adicional

Tem um script extra, `main_kfold.py`, no projeto.  
Ele faz quase a mesma coisa que o arquivo principal e está presente apenas para referência.

O `main_kfold.py` difere do `main.py` ao usar o KFold para configurar as rodadas, enquanto o
`main.py` usa o ShuffleSplit.  
O KFold decide automaticamente a fração da base de dados a ser
usada para validação e treino de acordo com a quantidade de cortes, enquanto o ShuffleSplit
nos permite definir valores arbitrários para as frações de treino e validação.

Usamos o ShuffleSplit no projeto para poder definir 10% para os testes e 90% para o treino.