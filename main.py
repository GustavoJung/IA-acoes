from re import X
import pandas_datareader as web
import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2022-10-08')

data = df.filter(['Close'])
dataset = data.values

training_data_len = math.ceil(len(dataset) * .8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, :]

x_train = [] #independente
y_train = [] #dependente

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0]) #60 valores de 0 a 59 e etc
    y_train.append(train_data[i, 0]) # 60

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape =(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=1, epochs=1)

test_data = scaled_data[training_data_len -60:, :]

x_test = []
y_test = dataset[training_data_len:, :] 
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i,0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(predictions - y_test ) ** 2)

train  = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(15,7))
plt.title('Modelo')
plt.xlabel('Data', fontsize=18) 
plt.ylabel('Preço em dolar', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#prever mais a frente
#buscar dados
apple_quote = web.dataReader('AAPL', data_source = 'yahoo', start='2012-01-01', end='2022-10-09')
#colocar os dados num dataframe
new_df = apple_quote.filter(['Close'])
#ultimos 60 valores e coloca-los num array
last_60_days = new_df[:60].values
#ajustar os dados para serem entre 0 e 1
last_60_days_scaled = scaler.transform(last_60_days)
#armazenar os valores no vetor X_test
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
#prever
pred_price = model.predict(X_test)
#voltar os dados de serem entre 0 e 1 para valor normal
pred_price = scaler.inverse_transform(pred_price)
