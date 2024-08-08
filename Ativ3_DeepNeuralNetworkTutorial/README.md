# Atividade 3 - Rede Neural Profunda, Tutorial

Lucas Carvalho Flores  
Ufal 2024


## Requisitos

* Python 3
* [Pipenv](https://pipenv.pypa.io/en/latest/)


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

Esta atividade está dividida em quatro scripts:

1. [Ativ31LinRegSingVar.py](Ativ31LinRegSingVar.py) implementa a primeira parte do tutorial, uma regressão linear com uma variável.
2. [Ativ32LinRegMultiVar.py](Ativ32LinRegMultiVar.py) implementa a segunda parte do tutorial, uma regressão lienar com múltiplas variáveis.
3. [Ativ33DeepNeuroNetSingVar.py](Ativ33DeepNeuroNetSingVar.py) implementa a terceira parte, uma rede neural com duas camadas ocultas e uma variável.
4. [Ativ34DeepNeuroNetMultiVar.py](Ativ34DeepNeuroNetMultiVar.py) implementa a quarta parte, uma rede neural com duas camadas ocultas e múltiplas variáveis.

O script [support.py](support.py) implementa algumas funções utilitárias, enquanto [compare.py](compare.py) executa os outros quatro scripts e exibe suas perdas para comparação.

O script [main.py](main.py) compila e treina uma rede neural baseada na de [Ativ34DeepNeuroNetMultiVar.py](Ativ34DeepNeuroNetMultiVar.py) para avaliar e prever os preços das casas em [Housing.csv](Housing.csv).

Para ver os detalhes da execução de algum dos scripts, como plotagens ou informação adicional no terminal, executar o script isoladamente. ie:
```bash
pipenv shell
python Ativ34DeepNeuroNetMultiVar.py
```
Isso vai executar a rede neural de múltiplas variáveis, escrevendo detalhes da execução no terminal e plotando as imagens para visualização.


## Referências

Tutorial da atividade - [Tensor Flow: Regressão Básica](https://www.tensorflow.org/tutorials/keras/regression)
