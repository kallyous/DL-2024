"""
Atividade 2 - Regressão Linear com Scikit-Learn.
Usa regressão linear e validação cruzada de 5 rodadas, em uma variável.
Usa ShuffleSplit para montar as rodadas/folds.
UFAL 2024
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit          # Rodadas de validação.
from sklearn.linear_model import LinearRegression         # Algoritmo de R.L.
from sklearn.metrics import mean_squared_error, r2_score  # Métricas


# Base de dados.
df = pd.read_csv('Housing.csv')

# Define quem são as variáveis dependentes e a variável dependente (alvo).
X = df[['area']]  # Nomes das colunas das variáveis independentes.
y = df['price']   # Nome da coluna alvo.

# Usa o ShuffleSplit para preparar as rodadas.
ss = ShuffleSplit(n_splits=5, test_size=0.1, random_state=42)

# Listas para armazenar os resultados.
mse_list = []
r2_list = []

# Iterar pelas rodadas de treino/teste e avaliar o modelo a cada passo.
fold = 1
for train_index, test_index in ss.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse_list.append(mse)
    r2_list.append(r2)

    print(f'Fold {fold}: MSE = {mse}, R² = {r2}')
    fold += 1

# Calcular as métricas médias.
mean_mse = np.mean(mse_list)
mean_r2 = np.mean(r2_list)

print(f'Mean MSE: {mean_mse}')
print(f'Mean R²: {mean_r2}')
