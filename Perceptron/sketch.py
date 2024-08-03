
import numpy as np


""" PROCEDIMENTOS """


def log(*args, **kwargs):
    """
    Função de log que imprime mensagens somente se DEBUG estiver definido como True.
    :param args: Argumentos posicionais passados para a função print.
    :param kwargs: Argumentos nomeados passados para a função print.
    """
    if DEBUG:
        print('\n', *args, sep='', **kwargs)


def predict(X: list, b: float, W: list) -> int:
    """Classificador binário de um vetor de variáveis.
    Args:
        b: Viés.
        X: Vetor de variáveis independentes.
        W: Vetor de pesos, com o peso do viés no último índice.
    Returns: Inteiro representando a encontrada.
    """

    # Calcula ativação:  a = Sum(x_i * w_i) + b*w_b
    activation = b * W[-1]
    for i in range(len(X)):
        activation += X[i] * W[i]

    # Retorna previsão baseado na ativação encontrada.
    return 1 if activation >= 0 else 0


def train(b: float, X: np.ndarray, Y: list,
          epoch: int, learn_rate: float) -> np.ndarray:
    """Treina os pesos com as configurações fornecidas.
    Args:
        b: Viés.
        X: Vetor de variáveis independentes normalizadas.
        Y: Vetor de rótulos.
        epoch: Quantidade de iterações de treino.
        learn_rate: Taxa de aprendizagem.
    Returns: Vetor W com os pesos das variáveis e do viés.
    """

    # Prepara vetor de pesos.
    rows, cols = X.shape
    W = np.ones(cols+1)

    # Laço das éguas.
    for e in range(1, epoch+1):
        # Redefine erro acumulado da época.
        error_sum = 0.0

        # Laço das linhas.
        for j in range(rows):

            pred = predict(X[j], b, W)  # Classifica.
            error = Y[j] - pred         # Calcula erro.
            error_sum += error**2       # Incrementa erro acumulado.

            # Calibra peso do viés.
            W[-1] += learn_rate * error

            # Calibra pesos das variáveis.
            for i in range(cols):
                W[i] += learn_rate * error * X[j][i]

        # Exibe progresso do treino.
        print(f'epoch={e}    learn_rate={learn_rate}    error={error_sum}')

    # Retorna pesos encontrados.
    return W


def evaluate(X: np.ndarray, b: float, W: list, Y: list):

    error_flat = 0
    rows, cols = X.shape

    for r in range(rows):
        if predict(X[r], b, W) != Y[r]:
            error_flat += 1

    return error_flat / rows


""" CONFIGURAÇÕES """

# Define previsão de float para 3 casas decimais.
np.set_printoptions(precision=3)

# Variável global para controle de debug
DEBUG = True

# x1 x2 x3 y
dataset = [
    [35, 95, 12, 0],
    [56, 21, 28, 0],
    [67, 67, 31, 0],
    [23, 23, 9, 0],
    [74, 89, 25, 0],
    [21, 67, 47, 1],
    [17, 98, 52, 1],
    [23, 56, 71, 1],
    [7, 12, 54, 1],
    [36, 34, 23, 1],
]

# Parâmetros do perceptron.
bias = 1.0
learning_rate = 0.05
epochs = 50


if __name__ == '__main__':


    """ PREPARO DOS DADOS """

    # Cópia dos dados.
    D = np.asarray(dataset)
    log('Dados:\n', D)

    # Separa variáveis independentes da variável dependente.
    X = D[:, :3].astype('float32')
    Y = D[:, 3:].flatten()
    log('X:\n', X)
    log('Y:\n', Y)

    # Normalizar X
    rows, cols = X.shape  # Número de linhas e colunas.
    S = np.zeros(cols)    # Vai segurar as somas de cada coluna.
    log(f'X.shape:  rows={rows}  cols={cols}')
    log('S:\n', S)

    # Soma todos os valores, guarda os totais de cada variável em S.
    for y in range(rows):
        for x in range(cols):
            S[x] += X[y][x]
    log('S:\n', S)

    # Agora vamos normalizar X usando S.
    for y in range(rows):
        for x in range(cols):
            X[y][x] = X[y][x] / S[x]
    log('X norm:\n', X)

    # Certifica corretude da normalização.
    S = np.zeros_like(S)
    for y in range(rows):
        for x in range(cols):
            S[x] += X[y][x]
    log('Sum.Feats.X:\n', S)


    """ TREINO """

    log('Treinando...')
    W = train(bias, X, Y, epochs, learning_rate)
    log('W:\n', W)


    """ AVALIAÇÃO """

    model_error = evaluate(X, bias, W, Y)
    log(f'Acurácia: {(1-model_error)*100} %')
