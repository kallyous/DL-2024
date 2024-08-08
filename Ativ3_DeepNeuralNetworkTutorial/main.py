'''
Previsão dos preços das casas com uma rede neural profunda.
Aplicação do tutorial em
    https://www.tensorflow.org/tutorials/keras/regression#regression_with_a_deep_neural_network_dnn .
Na base de dados da atividade.
Usar 'python <script>.py 2>/dev/null' se o TF tiver enxendo o saco com mensagens
de erro direto no terminal antes de passar por dentro do interpretador Python.
Lucas Carvalho Flores
'''


""" DEPENDÊNCIAS """

import os
import sys
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

'''Variável de ambiente que silencia o Tensor Flow.
Quando importado, o tensor flow está imprimindo diversas mensagens sobre o
estado da instalação do próprio tensorflow. Essas informações são importantes,
mas inúteis no contexto desse script.'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# Redefine o nível de log do TensorFlow para apenas reportar erros fatais.
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from tensorflow import keras
from tensorflow.keras import layers

# Assistente próprio para mensagens de log.
import support
from support import log, download



""" PROCEDIMENTOS """


def plot_loss_multi_var(history):
    """Plota o histórico de um treinamento (retornado por um Model.fit()).
    history.history[] para funcionar com histórico de um treino com
    multivariáveis.
    """

    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [price]')
    plt.legend()
    plt.grid(True)
    plt.show()


def build_and_compile_model(normalizer):
    """Ainda um modelo do tipo sequencial, esse terá duas camadas internas
    (ou ocultas) com função de ativação ReLU, e a camada de saída continua
    a mesma. No mais, o modelo continua funcionando como os anteriores, que
    eram apenas regressões lineares e buscavam otimizar (minimizar)
    a perda/loss.
    A primeira camanda (de entrada) se ocupa de normalizar os dados,
    eliminando assim a necessidade de normalizá-los antes de entregá-los
    pro modelo treinar/prever.
    """

    model = keras.Sequential([
        normalizer,
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.025))
    return model


""" Configurações """

# Controle de informação na saída.
if '--evaluate' in sys.argv:
    support.DEBUG = False
else:
    support.DEBUG = True

DEBUG = support.DEBUG

# Parâmetros de prints do Numpy.
np.set_printoptions(precision=2, suppress=True)

# Versão do Tensor Flow
log('\nTensor Flow', tf.__version__, '\n')

# Arquivo local da base de dados.
path_datafile = 'Housing.csv'


""" Carregamento dos Dados """

# Tipos de dados para as colunas.
cols_data_types = int

# Conversores das colunas categóricas para inteiros.
cols_convertes = {
    'mainroad': lambda x: 1 if x == 'yes' else 0,
    'guestroom': lambda x: 1 if x == 'yes' else 0,
    'basement': lambda x: 1 if x == 'yes' else 0,
    'hotwaterheating': lambda x: 1 if x == 'yes' else 0,
    'airconditioning': lambda x: 1 if x == 'yes' else 0,
    'prefarea': lambda x: 1 if x == 'yes' else 0,
    'furnishingstatus':
        lambda x: 2 if x == 'furnished' else (1 if x == 'semi-furnished' else 0)
}

raw_dataset = pd.read_csv(path_datafile,
                          dtype=cols_data_types,
                          converters=cols_convertes)

# Todas as colunas
cols_keep = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
             'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
             'parking', 'prefarea', 'furnishingstatus']

# Colunas descartadas
# cols_keep.remove('hotwaterheating')

dataset = raw_dataset[cols_keep]


log('\nDataset Inicial\n', dataset.tail(), sep='')
log('\nNot-a-Number por variável\n', dataset.isna().sum(), sep='')
log('\nDescrição dos dados\n', dataset.describe(), sep='')
log('\nInfo\n', dataset.info(), sep='')
log('\nMédia e desvio padrão mostram o quanto os intervalos variam\n',
    dataset.describe().transpose()[['mean', 'std']], sep='')
log('\nCorrelações\n', dataset.corr(), sep='')

# plt.figure(figsize=(8, 8))
# sns.heatmap(dataset.corr(), annot=True,
#             cmap='coolwarm', fmt='.2f', linewidths=.5)
# plt.show()


""" Limpeza dos dados """

# Remove entradas com colunas NaN.
dataset = dataset.dropna()

# Divisão dos dados em parte de treino e parte de testes.
train_dataset = dataset.sample(frac=0.8, random_state=0)
# TODO: Pesquisar o que random_state faz, bem como o que é.

test_dataset = dataset.drop(train_dataset.index)
# TODO: O que está havendo nesse drop(index)?


""" Inspeção dos Dados """

if DEBUG:
    sns.pairplot(test_dataset[['price', 'area', 'bedrooms', 'bathrooms']])
    plt.show()

log('\nEstatísticas de treino\n', train_dataset.describe().transpose())


""" Separar rótulo das variáveis """

train_features = train_dataset.copy()       # Faz cópia da parte de treino da base de dados.
train_labels = train_features.pop('price')  # Remove a variável rótulo.

test_features = test_dataset.copy()       # Faz cópia da parte de treino da base de dados.
test_labels = test_features.pop('price')  # Remove a variável rótulo.

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
    verbose=1,  # Setting verbose=0 supress logging.
    validation_split=0.2)  # Calculate validation results on 20% of the training data.

# Exibe resultado do treino.
if DEBUG:
    plot_loss_multi_var(model_fit_history)

# Coleta resultados para análise posterior.
test_results = model_dnn.evaluate(
    test_features,
    test_labels,
    verbose=0)

log('\nResultado:  loss = ', sep='', end='')
print(test_results)

# Linha extra ao final.
log()
