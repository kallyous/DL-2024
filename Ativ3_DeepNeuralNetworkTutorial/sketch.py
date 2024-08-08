import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras.optimizers import Adam


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

# Carrega a porra toda
df = pd.read_csv('Housing.csv',
                          dtype=cols_data_types,
                          converters=cols_convertes)

# Separar características e alvo
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
        'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
        'parking', 'prefarea', 'furnishingstatus']]
y = df['price']

# Camada de normalização
normalizer = Normalization(axis=-1)

# Adequar o normalizador às variáveis de treino
normalizer.adapt(np.array(X))

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir o modelo Sequential
model = Sequential()
model.add(normalizer)
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compilar o modelo
model.compile(optimizer=Adam(0.25), loss='mean_squared_error')

# Treinar o modelo
history = model.fit(X_train, y_train,
                    epochs=100, validation_split=0.2, verbose=0)

# Loss
loss = model.evaluate(X_test, y_test, verbose=0)

# Fazer previsões
y_pred = model.predict(X_test)

# Calcular e exibir métricas
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Função para calcular MAPE
# def mean_absolute_percentage_error(y_true, y_pred):
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#
# mape = mean_absolute_percentage_error(y_test, y_pred)

# Exibir as métricas
print(f"Loss: {loss}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")
# print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
