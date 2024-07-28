'''
Regressão linear com uma variável.
Primeira parte do tutorial em https://www.tensorflow.org/tutorials/keras/regression .
Algumas correções foram feitas.
Usar 'python <script>.py 2>/dev/null' se o TF tiver enxendo o saco com mensagens de erro direto no terminal
sem passar por dentro do interpretador do Python.

Lucas Carvalho Flores
'''

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

'''Variável de ambiente que silencia o Tensor Flow.
Quando importado, o tensor flow está imprimindo diversas mensagens sobre o estado da instalação 
do próprio tensorflow. Essas informações são importantes, mas inúteis no contexto desse script.'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# Redefinir o nível de log do TensorFlow para erros fatais apenas.
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from tensorflow import keras
from tensorflow.keras import layers

# Assistente prórpio para mensagens de log.
import support
from support import log, download


""" PROCEDIMENTOS """

def plot_loss(history):
    """Plota o histórico de um treinamento (retornado por um Model.fit()).
    Ao invés de history.history[] usa apenas history[], para funcionar com
    histórico de um treino com uma variável de treino apenas.
    """
    plt.cla()
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_horsepower(x, y):
    plt.cla()
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()
    plt.show()


""" Configurações """

# Controle de informação na saída.
if '--evaluate' in sys.argv:
    support.DEBUG = False
else:
    support.DEBUG = True

DEBUG = support.DEBUG

np.set_printoptions(precision=2, suppress=True)   # Parâmetros de visualização/prints do Numpy.
log('\nTensor Flow', tf.__version__, '\n')  # Versão do Tensor Flow

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
path_datafile = 'auto-mpg.csv'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']


""" Carregamento dos Dados """

download(url, path_datafile)

raw_dataset = pd.read_csv(path_datafile, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
log( '\nDataset Tail\n', dataset.tail() )


""" Limpeza dos dados """

log( '\nIs NA\n', dataset.isna().sum(), sep='' )
dataset = dataset.dropna()

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
log( '\nDataset Tail - pós pd.get_dummies(..., columns=[\'Origin\'], ...)\n', dataset.tail() )

# Divisão dos dados em parte de treino e parte de testes.
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Inspeção dos dados.
if DEBUG:
    sns_plot = sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
    # sns_plot.figure.savefig('auto-mpg-lin-reg-single-var.png')  # Salvar imagem em arquivo.
    plt.show()  # Exibir plotagem em nova janela.
log( '\nEstatísticas de treino\n', train_dataset.describe().transpose())

# Separar rótulo das variáveis.

train_features = train_dataset.copy()
train_labels = train_features.pop('MPG')

test_features = test_dataset.copy()
test_labels = test_features.pop('MPG')


# 4 - Normalização

# Observe os intervalos das variáveis e o quanto elas variam, mostrando a necessidade de normalização das variáveis.
log( '\nMédia e desvio padrão mostram o quanto os intervalos variam\n',
     train_dataset.describe().transpose()[['mean', 'std']], sep='' )

# Camada de normalização. É uma ferramenta do Keras para não precisarmos normalizar tudo à mão.
normalizer = tf.keras.layers.Normalization(axis=-1)

# Adequa a camada de normalização às variáveis de treino.
normalizer.adapt(np.array(train_features))

# Calcula média e variância, e os armazena na camada de normalização.
log( '\n', normalizer.mean.numpy(), sep='' )

# Tensor flow não aceita tipos inteiros. Antes de passarmos coisa para o TF processar, converter tudo em float.
# https://stackoverflow.com/questions/58636087/tensorflow-valueerror-failed-to-convert-a-numpy-array-to-a-tensor-unsupporte
train_features = train_features.astype(np.float32)
log( '\nTrain Feat <- train_features.astype(np.float32)\n', train_features.tail(), '\n', sep='')

# When the layer is called, it returns the input data, with each feature independently normalized:
first = np.array(train_features[:1])
with np.printoptions(precision=2, suppress=True):
  log('First example:', first, '\n')
  log('Normalized:', normalizer(first).numpy())


# 5 - Regressão Linear
# https://www.tensorflow.org/tutorials/keras/regression#linear_regression

# Faz uma array básica a partir da ndarray com a feature a ser usada nos treinos, para normalizá-la.
horsepower = np.array(train_features['Horsepower'])

# Instantiate the tf.keras.layers.Normalization and fit its state to the horsepower data
horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

# Build the Keras Sequential model
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

# tensorflow.keras.Sequential.summary() manda direto pro stdout ao invés de retornar uma string.
if DEBUG:
    horsepower_model.summary()

''' This model will predict 'MPG' from 'Horsepower'.
Running the untrained model on the first 10 'Horsepower' won't yield good results,
but it has the expected shape of (10, 1). '''
horseshit = horsepower_model.predict(horsepower[:10], verbose=0)
log('\nTeste do modelo:\n', horseshit, sep='')

'''Once the model is built, configure the training procedure using the Keras Model.compile method.
The most important arguments to compile are the loss and the optimizer, since these define what
 will be optimized (mean_absolute_error) and how (using the tf.keras.optimizers.Adam).'''
horsepower_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

# Use Keras Model.fit to execute the training for 100 epochs.
history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    verbose=0,               # Suppress progress output.
    validation_split = 0.2)  # Calculate validation results on 20% of the training data.

# Visualize the model's training progress using the stats stored in the history object.
df_hist = pd.DataFrame(history.history)
df_hist['epoch'] = history.epoch
log( '\n', df_hist.tail(), sep='')

if DEBUG:
    plot_loss(df_hist)

# Collect the results on the test set for later
test_results = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)

if DEBUG:
    # Since this is a single variable regression, it's easy to view the model's predictions as a function of the input.
    x = tf.linspace(0.0, 250, 251)
    y = horsepower_model.predict(x, verbose=0)
    plot_horsepower(x, y)

log( '\nResultado:  loss = ', sep='', end='' )
print(test_results)

# Nova linha extra ao final.
log()
