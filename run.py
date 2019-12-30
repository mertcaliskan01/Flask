import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from pandas_datareader import data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
from keras.layers import Activation, Dense
from Predict.build_timeseries import build_timeseries
from Predict.trim_dataset import trim_dataset

_BATCH_SIZE = 75
_TIME_STEPS = 60
_EPOCHS = 10
_NAMES_OF_THE_STOCKS = ['AKBNK.IS', 'ASELS.IS', 'YKBNK.IS', 'SISE.IS', 'THYAO.IS']
TRAINING_COLUMNS = ['High', 'Low', 'Open', 'Close', 'Volume']
START_DATE = '2010-01-01'
END_DATE   = '2019-12-05'
OUTPUT_PATH = './logs/'

def train(NAMES_OF_THE_STOCKS,START_DATE,END_DATE,BATCH_SIZE,TIME_STEPS,EPOCHS):
    # Fetching the data
    panel_data = data.DataReader(NAMES_OF_THE_STOCKS, 'yahoo', START_DATE, END_DATE)

    # Getting only close prices
    close = panel_data['Close']

    # Getting all weekdays between 01/01/2000 and 05/12/2019
    all_weekdays = pd.date_range(start=START_DATE, end=END_DATE, freq='B')


    close = close.reindex(all_weekdays)
    panel_data_reindexed = panel_data.reindex(all_weekdays)


    close = close.fillna(method='ffill')
    panel_data_reindexed = panel_data_reindexed.fillna(method='ffill')

    # Split and normalize data
    df_train, df_test = train_test_split(panel_data_reindexed, train_size=0.8, test_size=0.2, shuffle=False)
    x = df_train.loc[:,TRAINING_COLUMNS].values
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x)
    x_test = min_max_scaler.transform(df_test.loc[:,TRAINING_COLUMNS])


    # Let's form train, validation and test sets
    x_train, y_train = build_timeseries(x_train, 3, TIME_STEPS) # Close column is at index 3
    x_train = trim_dataset(x_train, BATCH_SIZE)
    y_train = trim_dataset(y_train, BATCH_SIZE)
    x_temp, y_temp = build_timeseries(x_test, 3, TIME_STEPS)
    x_validation, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE), 2)
    y_validation, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE), 2)


    # Create the model
    lstm_model = Sequential()

    lstm_model.add(LSTM(50, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_train.shape[2]), return_sequences=True))
    lstm_model.add(Dropout(0.5))

    lstm_model.add(LSTM(100, return_sequences=False))
    lstm_model.add(Dropout(0.5))

    lstm_model.add(Dense(1))
    lstm_model.add(Activation('linear'))
    lstm_model.compile(loss='mse', optimizer='adam')
    
    # Train the model
    csv_logger = CSVLogger('train.log') #append=True)

    history = lstm_model.fit(x_train, y_train, epochs=EPOCHS, verbose=2, batch_size=BATCH_SIZE,
                        shuffle=False, validation_data=(trim_dataset(x_validation, BATCH_SIZE),
                        trim_dataset(y_validation, BATCH_SIZE)), callbacks=[csv_logger])

    # Visualize training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
    # Predict
    y_prediction = lstm_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
    y_prediction = y_prediction.flatten()
    y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
    
    graph_datas = [y_prediction, y_test_t]
    return graph_datas

def show_graph(name,Result):
    plt.figure()
    plt.plot(Result[0])
    plt.plot(Result[1])
    plt.title(name + ' Prediction vs Real Stock Price')
    plt.ylabel('Price')
    plt.xlabel('Days')
    plt.legend(['Prediction', 'Real'], loc='upper left')
    plt.show()

akbankResult = train(_NAMES_OF_THE_STOCKS[0], START_DATE, END_DATE, _BATCH_SIZE, _TIME_STEPS, _EPOCHS)
aselsanResult = train(_NAMES_OF_THE_STOCKS[1], START_DATE, END_DATE, _BATCH_SIZE, _TIME_STEPS, _EPOCHS)
yapıkrediResult = train(_NAMES_OF_THE_STOCKS[2], START_DATE, END_DATE, _BATCH_SIZE, _TIME_STEPS, _EPOCHS)
siseResult = train(_NAMES_OF_THE_STOCKS[3], START_DATE, END_DATE, _BATCH_SIZE, _TIME_STEPS, _EPOCHS)
thyResult = train(_NAMES_OF_THE_STOCKS[4], START_DATE, END_DATE, _BATCH_SIZE, _TIME_STEPS, _EPOCHS)

show_graph("Akbank",akbankResult)
print("--------PREDICTION----------")
print(akbankResult[0])
print("----------REAL----------")
print(akbankResult[1])

show_graph("Aselsan",aselsanResult)
print("--------PREDICTION----------")
print(aselsanResult[0])
print("----------REAL----------")
print(aselsanResult[1])

show_graph("Yapı Kredi",yapıkrediResult)
print("--------PREDICTION----------")
print(yapıkrediResult[0])
print("----------REAL----------")
print(yapıkrediResult[1])

show_graph("Şişecam",siseResult)
print("--------PREDICTION----------")
print(siseResult[0])
print("----------REAL----------")
print(siseResult[1])

show_graph("THY",thyResult)
print("--------PREDICTION----------")
print(thyResult[0])
print("----------REAL----------")
print(thyResult[1])