from PIL import Image
import streamlit as st
from constants import *
from ai_improver import *
from cv_scanner import *

import json
import datetime;
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
    ax.title('LSTM')
    ax.xlabel('Epochs')
    ax.ylabel('MSE')
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

    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym=ETH&tsym=USD&limit=500')
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    target_col = 'close'

    hist.drop(["conversionType", "conversionSymbol"], axis = 'columns', inplace = True)

    ct = datetime.datetime.now()
    st.write("Current time:", ct)
    st.header('Bitcoin daily activity')
    st.write(hist)
    
    train, test = train_test_split(hist, test_size=0.2)

    st.header('Closing daily price')
    line_plot(train[target_col], test[target_col], 'training', 'test', title='')

    np.random.seed(42)
    window_len = 10
    test_size = 0.2
    zero_base = True
    lstm_neurons = 100
    epochs = 30
    batch_size = 32
    loss = 'mse'
    dropout = 0.2
    optimizer = 'adam'

    train, test, X_train, X_test, y_train, y_test = prepare_data(
        hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)

    st.write('Train shape: ' + str(train.shape))
    st.write('Test shape: ' + str(test.shape))
    st.write('X_Train shape: ' + str(X_train.shape))
    st.write('y_Train shape: ' + str(y_train.shape))
    st.write('X_Test shape: ' + str(X_test.shape))
    st.write('y_Test shape: ' + str(y_test.shape))

    model = build_lstm_model(
        X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        optimizer=optimizer)
    history = model.fit(
        X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

    line_plot2(history.history['loss'], history.history['val_loss'], 'Train loss', 'Validation loss', title='')


    image = Image.open('resume_image.jpeg')
    st.image(image, caption='Photo by Unseen Studio on Unsplash')
    st.header('Improving your CV in seconds using ChatGPT!')
    st.write('This app is meant to improve the quality of your CV by using Artificial Intelligence\n Start by downloading the template, fill the information, upload your CV and enjoy the magic! :smile:')
    st.write("\n Let's see what you got! Download the following template and fill it out with your information! :sunglasses:")
#    download_template()
    uploaded_file = st.file_uploader("Upload your CV here! :point_down:") 
    reviewed_experiences = []
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write(stringio)
        string_data = stringio.read()
        st.subheader('Lets discuss the summary :male-detective:')
        reviewed_summary = summary_result(string_data)
        st.subheader('Lets discuss the work experience :office:')
        experiences = experience_parser(string_data)
        st.write('We noticed that you added ' + str(len(experience_parser(string_data)))+ ' work experiences! :eyes:')
        for e in experiences:
            print('Experience:')
            print(e.split('[SEP]')[-2])
            review_experience = experience_result(e.split('[SEP]')[-2])
            reviewed_experiences.append(review_experience)
        new_file = open('cv_improved.txt','w+')
        new_file.write('SUMMARY:\n')
        new_file.write(reviewed_summary)
        for e in range(len(reviewed_experiences)):
            new_file.write('\nEXPERIENCE %i \n'%(e))
            new_file.write(reviewed_experiences[e])
        new_file.close()
        download_result()
        

    
