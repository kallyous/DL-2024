'''
Rede Neural Profunda - Multi Variables
Terceira parte do tutorial em
    https://www.tensorflow.org/tutorials/keras/regression#regression_with_a_deep_neural_network_dnn .
Algumas correções foram feitas.
Usar 'python <script>.py 2>/dev/null' se o TF tiver enxendo o saco com mensagens de erro direto no terminal
sem passar por dentro do interpretador do Python.
Lucas Carvalho Flores
'''

''' DEPENDÊNCIAS '''

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



'''Variável de ambiente que silencia o Tensor Flow.
Quando importado, o tensor flow está imprimindo diversas mensagens sobre o
estado da instalação do próprio tensorflow. Essas informações são importantes,
mas inúteis no contexto desse script.'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf



# Redefinir o nível de log do TensorFlow para erros fatais apenas.
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from tensorflow import keras
from tensorflow.keras import layers

# Assistente próprio para mensagens de log.
import support
from support import log, download



""" PROCEDIMENTOS """


def plot_loss(history):
    """Plota o histórico de um treinamento (retornado por um Model.fit()).
    history.history[] para funcionar com histórico de um treino com
    multivariáveis.
    """
    plt.cla()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()


def build_and_compile_model(normalizer):
    """Ainda um modelo do tipo sequencial, esse terá duas camadas internas
    (ou ocultas) com função de ativação ReLU, e a camada de saída continua
    a mesma. No mais, o modelo continua funcionando como os anteriores, que
    eram apenas regressões lineares e buscavam otimizar (minimizar)
    a perda/loss.
    """
    model = keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


""" Configurações """

# Controle de informação na saída.
if '--evaluate' in sys.argv:
    support.DEBUG = False
else:
    support.DEBUG = True

DEBUG = support.DEBUG

np.set_printoptions(precision=2,
                    suppress=True)  # Parâmetros de prints do Numpy.
log('\nTensor Flow', tf.__version__, '\n')  # Versão do Tensor Flow

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'  # Url da base de dados.
path_datafile = 'auto-mpg.csv'  # Arquivo local da base de dados.
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                # Nomes das colunas.
                'Weight', 'Acceleration', 'Model Year', 'Origin']

""" Carregamento dos Dados """

download(url, path_datafile)

raw_dataset = pd.read_csv(path_datafile, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()

log('\nDataset Inicial\n', dataset.tail(), sep='')
log('\nNot-a-Number por variável\n', dataset.isna().sum(), sep='')
log('\nDescrição dos dados\n', dataset.describe(), sep='')
log('\nMédia e desvio padrão mostram o quanto os intervalos variam\n',
    dataset.describe().transpose()[['mean', 'std']], sep='')

""" Limpeza dos dados """

# Remove entradas com colunas NaN.
dataset = dataset.dropna()

# Mapeia variável categórica 'Origin' em números.
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
log('\nOrigin <- map() : categorias convertidas para números\n',
    dataset.tail(), sep='')

# Transforma a coluna 'Origin' de valores discretos em colunas boleanas.
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
log('\nOrigin <- dummies : Origin vira colunas boleanas baseadas nos valores contidos\n',
    dataset.tail())

# Divisão dos dados em parte de treino e parte de testes.
train_dataset = dataset.sample(frac=0.8, random_state=0)
# TODO: Pesquisar o que random_state faz, bem como o que é.

test_dataset = dataset.drop(train_dataset.index)
# TODO: O que está havendo nesse drop(index)?


""" Inspeção dos Dados """

if DEBUG:
    sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']],
                 diag_kind='kde')
    plt.show()

log('\nEstatísticas de treino\n', train_dataset.describe().transpose())


""" Separar rótulo das variáveis """

train_features = train_dataset.copy()     # Faz cópia da parte de treino da base de dados.
train_labels = train_features.pop('MPG')  # Remove a variável rótulo.

test_features = test_dataset.copy()     # Faz cópia da parte de treino da base de dados.
test_labels = test_features.pop('MPG')  # Remove a variável rótulo.

# Ajuste importante para o Tensor Flow
'''Tensor flow não aceita tipos inteiros.
Antes de passarmos coisas para o TF processar, converter variáveis de treino em float.
https://stackoverflow.com/questions/58636087/tensorflow-valueerror-failed-to-convert-a-numpy-array-to-a-tensor-unsupporte
'''
train_features = train_features.astype(np.float32)
log('\ntrain_features.astype(np.float32)\n',
    train_features.tail(), '\n', sep='')


""" Normalização """

# Camada de normalização.
# É uma ferramenta do Keras para não precisarmos normalizar tudo à mão.
normalizer = tf.keras.layers.Normalization(axis=-1)

# Adequa normalizador às variáveis de train_features.
normalizer.adapt(np.array(train_features))

log('\nNormalization Means\n', normalizer.mean.numpy(), sep='')


""" Rede Neural Profunda com Multi Variáveis """

# Configura e compila o modelo usando a função definida na seção PROCEDIMENTOS.
model_dnn = build_and_compile_model(normalizer)

# Modelo construído. Sequential.summary() manda direto pra stdout.
if DEBUG:
    model_dnn.summary()

log('\nTreinando modelo por 100 épocas...')
# Treina modelo com Model.fit() por 100 épocas.
model_fit_history = model_dnn.fit(
    train_features,
    train_labels,
    epochs=100,
    verbose=0,  # Setting verbose=0 supress logging.
    validation_split=0.2)  # Calculate validation results on 20% of the training data.

# Exibe resultado do treino.
if DEBUG:
    '''Esse modelo é ligeiramente melhor que a regressão linear de uma variável.'''
    plot_loss(model_fit_history)

# Coleta resultados para análise posterior.
test_results = model_dnn.evaluate(
    test_features,
    test_labels,
    verbose=0)

log('\nResultado:  loss = ', sep='', end='')
print(test_results)

# Linha extra ao final.
log()
