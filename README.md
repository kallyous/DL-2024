# DL-2024


Atividades da disciplina de Redes Neurais.

Lucas Carvalho Flores  
Ufal 2024


## Atividades

[Ativ 1 - Perceptron (Local)](Task0_Perceptron)

[Ativ 2 - Regressão Linear (Local)](Ativ2_LinearRegression/)

[Ativ 3 - Regressão Linear com Rede Neural Profunda (Local)](Ativ3_DeepNeuralNetworkTutorial/)


## Requerimentos & Instalação

#### TL; DR

```shell
pip install --user pipenv    # Instala o Pipenv
pipenv install               # Instala dependências
pipenv shell                 # Ativa ambiente virtual
cd <Pasta-da-Atividade>      # Entra no diretório da atividade
python <script-desejado>.py  # Executa script desejado
```

#### Detalhes sobre os requerimentos e instalação

Atividades feitas localmente (e não no Colab) usam o [Pipenv](https://pipenv.pypa.io/en/latest/) para gerenciamento de ambiente virtual e dependências.

* Python 3
* [Pipenv](https://pipenv.pypa.io/en/latest/)

**1.** Instalar o [Pipenv](https://pipenv.pypa.io/en/latest/) usando Pip:

```shell
pip install --user pipenv    # Instala o Pipenv
```

**2.** No diretório raiz do projeto, executar:
```bash
pipenv install               # Instala dependências
pipenv shell                 # Atiba ambiente virtual
```

**3.** Agora todas as dependências estão instaladas no ambiente virtual da raiz do projeto. Entre na pasta da atividade desejada e execute o script python normalmente.

```
cd <Pasta-da-Atividade>      # Entra no diretório da atividade
python <script-desejado>.py  # Executa script desejado
```

Exemplo:

```
pip install --user pipenv

pipenv install
pipenv shell

cd Ativ3_DeepNeuralNetworkTutorial
python main.py
```



### Tasks do repo da disciplina

Aqui estão as tarefas propostas no repositório da disciplina, feitas como eu entendi que era o esperado.

Repo: [https://github.com/kallyous/deep-learning/tree/main/tasks](https://github.com/kallyous/deep-learning/tree/main/tasks)
(Sim, é um fork pessoal do repo do professor)

[Tarefa 0](Task0_Perceptron) - Perceptron feito com Numpy, sem framework.

[Tarefa 1](Task1_LogisticRegression) - Regressão Logística feita com numpy, sem framework.
