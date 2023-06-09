from PIL import Image
import streamlit as st
from constants import *

import json
import math
import datetime
from time import sleep
import requests
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data
    
def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    st.pyplot(fig)

def line_plot2(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1,'r',linewidth=2, label=label1)
    ax.plot(line2, 'g',linewidth=2, label=label2)
    ax.set_title('LSTM', fontsize=16)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('MSE', fontsize=14)
    ax.legend(loc='best', fontsize=16)
    st.pyplot(fig)

def normalise_zero_base(df):
    return df / df.iloc[0] - 1

def normalise_min_max(df):
    return (df - df.min()) / (data.max() - df.min())
    
def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)
    
def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    train_data, test_data = train_test_split(df, test_size=test_size)
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values

    if data_choice == 'Yes':
        st.sidebar.write('train_data shape:' + str(train_data.shape))
        st.sidebar.write('test_data shape:' + str(test_data.shape))
        st.sidebar.write('X_train shape:' + str(X_train.shape))
        st.sidebar.write('X_test shape:' + str(X_test.shape))
        st.sidebar.write('y_train shape:' + str(y_train.shape))
        st.sidebar.write('y_test shape:' + str(y_test.shape))

    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, y_train, y_test
    
def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear',
                     dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model



if __name__=='__main__':

    st.sidebar.title('Navigation')

    ct = datetime.datetime.now()
    st.sidebar.write("Current time:", ct)

    data_choice = st.sidebar.radio("Show data",("No", "Yes"))
    coin_choice = st.sidebar.radio("Coin",("BTC", "ETH"))

    st.sidebar.write("Data Parameters")
    days = st.sidebar.radio("Days",(500,1000))
    test_size = st.sidebar.radio("Test Size",(0.1,0.2))
    window_len = st.sidebar.radio("Window Length",(5,10,20))

    st.sidebar.write("LSTM Parameters")
    batch_size = st.sidebar.radio("Batch Size",(8,16,32))
    lstm_neurons = st.sidebar.radio("Neuron Count",(50,100,200))
    epochs = st.sidebar.radio("Epochs",(15,30,60))
    dropout = st.sidebar.radio("Dropout",(0.1,0.2,0.3))
    optimizer = st.sidebar.radio("Optimizer",("adam"))
    loss = st.sidebar.radio("Loss",("mse"))

    np.random.seed(42)
    zero_base = True

    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym='+coin_choice+'&tsym=USD&limit='+str(days))
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    target_col = 'close'

    hist.drop(["conversionType", "conversionSymbol"], axis = 'columns', inplace = True)

    if data_choice == 'Yes':
        st.header(coin_choice +' daily activity')
        st.write(hist.sort_values(by=['time'], ascending=False))
    
    train, test = train_test_split(hist, test_size=test_size)

    st.header(coin_choice+' Closing daily price')
    line_plot(train[target_col], test[target_col], 'training', 'test', title='')

    train, test, X_train, X_test, y_train, y_test = prepare_data(
        hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)

    model = build_lstm_model(
        X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        optimizer=optimizer)
    history = model.fit(
        X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

    line_plot2(history.history['loss'], history.history['val_loss'], 'Train loss', 'Validation loss', title='')

    preds = model.predict(X_test).squeeze()
 
    st.sidebar.write('mae = ' + str(mean_absolute_error(preds, y_test)))
    st.sidebar.write('mse = ' + str(mean_squared_error(preds, y_test)))
    st.sidebar.write('r2 = ' + str(r2_score(y_test, preds)))

    if data_choice == 'Yes':
        st.write('First test of window shape:' + str(test[target_col].values[:-window_len].shape))
        st.write(test[target_col].values[:-window_len])

    targets = test[target_col][window_len:]
    preds = test[target_col].values[:-window_len] * (preds + 1)

    if data_choice == 'Yes':
        st.write('Target shape:' + str(targets.shape))
        st.write(targets)

        st.write('(Normalized) y_test shape:' + str(y_test.shape))
        st.write(y_test)

        st.write('Predicted shape:' + str(preds.shape))
        st.write(preds)

    preds = pd.Series(index=targets.index, data=preds)

    print('rmse = ' + str(math.sqrt(mean_squared_error(preds, targets))))
    
    line_plot(targets, preds, 'actual', 'prediction', lw=3)


    progress_text = "Waiting 100 seconds to refresh:"
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        sleep(1)
        my_bar.progress(percent_complete/100, text=progress_text)



