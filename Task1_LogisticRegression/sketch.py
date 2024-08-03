import numpy as np
from numpy import ndarray
from math import exp



""" PROCEDIMENTOS """


def log(*args, **kwargs):
    """
    Função de log que imprime mensagens somente se DEBUG estiver definido como True.
    :param args: Argumentos posicionais passados para a função print.
    :param kwargs: Argumentos nomeados passados para a função print.
    """
    if DEBUG:
        print('\n', *args, sep='', **kwargs)


def predict(X: ndarray, C: ndarray) -> float:
    """Classificação com regressão logística.
    Vulgo 'regressão linear, mas com função sigmoide na ativação'.
    Args:
        X: Vetor de atributos.
        C: Vetor de coeficientes (ou pesos), contendo o coeficiente do viés
           no último índice.
    Returns: Número real no intervalo [0, 1].
    """

    # Começamos adicionando direto o coeficiente do viés.
    y_hat = C[-1]

    # Incrementando y_hat com o produtório dos coeficientes pelos atributos.
    for i in range(len(X)):
        y_hat += C[i] * X[i]

    # Sigmóide bonito e formoso.
    # The math.exp() method returns E raised to the power of x (E^x).
    return 1.0 / (1.0 + exp(-y_hat))


'''
# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            error = row[-1] - yhat
            sum_error += error**2
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return coef
'''


# Estima/recalcula coeficientes da regressão logística usando
# descida estocástica de gradiente.
def train_coefficients_sgd(X: ndarray, Y: list,
                           learn_rate: float, epochs: int) -> ndarray:
    """Treina modelo e encontra os coeficientes usando stochastic gradient
    descent.
    Args:
        X: Vetor de linhas de atributos.
        Y: Vetor de rótulos.
        learn_rate: Taxa de aprendizagem.
        epochs: Quantidade de épocas.
    Returns: Vetor com os coeficientes do modelo.
    """

    # Pega as dimensões de X.
    rows, cols = X.shape

    # Inicia coeficientes com 0.0.
    C = np.zeros(cols+1)  # O último coeficiente é do intercept/bias.

    # Laço das eras.
    for epoch in range(1, epochs+1):
        error_sum = 0.0

        # Iteração sobre todas as linhas de X.
        for i in range(rows):

            y_hat = predict(X[i], C)  # Classifica X[i] com coeficientes atuais.
            error = Y[i] - y_hat      # Calcula erro da linha atual.
            error_sum += error**2     # Incrementa erro acumulado.

            # Atualiza coeficiente do intercept/bias.
            C[-1] = C[-1] + learn_rate * error * y_hat * (1.0 - y_hat)

            # Laço para atualizar todos os outros coeficientes.
            for j in range(cols):

                # Stochastic Gradient Descent
                C[j] = C[j] + learn_rate * error * y_hat * (1.0 - y_hat) * X[i][j]

        # Info sobre progresso.
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learn_rate, error_sum))

    # Retorna vetor de coeficientes encontrado.
    return C



""" CONFIGURAÇÕES """

# Define previsão de float para 3 casas decimais.
np.set_printoptions(precision=3)

# Variável global para controle de debug
DEBUG = True

# Dataset fictício pra ir testando as coisas enquanto desenvolvo.
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]

# Coeficientes prontos para o mesmo propósito.
coefficients = [0.852573316, -1.104746259, -0.406605464]

# Testa predict()
print('Testando predict():')
for row in dataset:
    pred = predict(row[:2], coefficients)
    print(f'Expected={row[-1]}, Predicted={pred}, {round(pred)}')

# Prepara os dados.
dataset = np.asarray(dataset)
X = dataset[:, :2]
Y = dataset[:, 2:]
print('Dataset:\n', dataset)
log('X:\n', X)
log('Y:\n', Y)

coefficients = train_coefficients_sgd(X, Y, 0.1, 100)

print('Testando coeficientes encontrados:')
for row in dataset:
    pred = predict(row[:2], coefficients)
    print(f'Expected={row[-1]}, Predicted={pred}, {round(pred)}')

print(coefficients)
