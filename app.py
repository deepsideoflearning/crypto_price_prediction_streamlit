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

if __name__=='__main__':

    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym=ETH&tsym=USD&limit=500')
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    target_col = 'close'

    hist.drop(["conversionType", "conversionSymbol"], axis = 'columns', inplace = True)

    ct = datetime.datetime.now()
    st.write("Current time:-", ct)
    st.header('Bitcoin daily activity')
    st.write(hist)
    
    train, test = train_test_split(hist, test_size=0.2)
    line_plot(train[target_col], test[target_col], 'training', 'test', title='')

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
        

    
