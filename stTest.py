'''
    streamlit run stTest.py
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TF to use only the CPU
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
import datetime
from datetime import date



def ts_train_test_normalize(all_data,time_steps,for_periods):
    '''
    input: 
      data: dataframe with dates and price data
    output:
      X_train, y_train: data from 2013/1/1-2018/12/31
      X_test:  data from 2019 -
      sc:      insantiated MinMaxScaler object fit to the training data
      
    '''   
    import numpy as np
    from datetime import date

    today = date.today()
    
    if(today.day+4>30):
        target_day= 3
        target_month = today.month+1
    else:
        target_day= today.day+7
        target_month = today.month

    
    # Sort the Data
    hd = all_data.copy()
    df = hd.sort_values('Date')
    df.set_index('Date', inplace=True)
    
    # Slice the Data
    ts_From = datetime.datetime(today.year,today.month,today.day)
    ts_To   = datetime.datetime(today.year,target_month,target_day)
    
    tr_From = datetime.datetime(2017, 1, 1)
    tr_To   = datetime.datetime(today.year,today.month,today.day)
    
    ts_test = df.loc[ts_From:ts_To,:].iloc[:,0:1].values
    ts_train = df.loc[tr_From:tr_To,:].iloc[:,0:1].values


    ts_train_len = len(ts_train)
    ts_test_len = 7
    

    # scale the data
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0,1))
    ts_train_scaled = sc.fit_transform(ts_train)


    # create training data of s samples and t time steps
    X_train = []
    y_train = []
    y_train_stacked = []
    for i in range(time_steps,ts_train_len-1): 
        X_train.append(ts_train_scaled[i-time_steps:i,0])
        y_train.append(ts_train_scaled[i:i+for_periods,0])
    X_train, y_train = np.array(X_train), np.array(y_train)

   
    # Reshaping X_train for efficient modelling
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))


    inputs = pd.concat((df.loc[tr_From:tr_To,:]["Adj Close"], df.loc[ts_From:ts_To,:]["Adj Close"]),axis=0).values
    inputs = inputs[len(inputs) - ts_test_len - time_steps:]
    inputs = inputs.reshape(-1,1)
    inputs  = sc.transform(inputs)
        
        
    # Preparing X_test
    X_test = []
    for i in range(time_steps,ts_test_len+time_steps-for_periods):
        X_test.append(inputs[i-time_steps:i,0])
      
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
    
    
    return X_train, y_train , X_test, sc


         

def LSTM_model(X_train, y_train, X_test, sc, for_periods):
    # create a model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM
    from tensorflow.keras.optimizers import SGD
    
    # The LSTM architecture
    my_LSTM_model = Sequential()
    my_LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    #my_LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    #my_LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    my_LSTM_model.add(LSTM(units=50, activation='tanh'))
    my_LSTM_model.add(Dense(units=for_periods))

    # Compiling
    my_LSTM_model.compile(optimizer=SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=False),loss='mean_squared_error')
    # Fitting to the training set
    my_LSTM_model.fit(X_train,y_train,epochs=50,batch_size=150, verbose=0)

    LSTM_prediction = my_LSTM_model.predict(X_test)
    LSTM_prediction = sc.inverse_transform(LSTM_prediction)

    return my_LSTM_model, LSTM_prediction


def actual_pred_plot(preds, dff):
    from tensorflow.keras.metrics import MeanSquaredError    

    today = date.today()

    
    # Sort the Data
    hd = dff.copy()
    df = hd.sort_values('Date')
    df.set_index('Date', inplace=True)

    # Slice the Data
    tr_From = datetime.datetime(today.year,today.month-1,today.day)
    tr_To   = datetime.datetime(today.year,today.month,today.day)

    ts_train = df.loc[tr_From:tr_To,:].iloc[:,0:1].values
    
    ts_train = [item for sublist in ts_train for item in sublist]
    
    
    ts = ts_train
    ts1 = ts_train + preds[:,0].tolist()

    #print(preds[:,0].tolist())

    t = np.arange(0, 36, 36/len(ts_train + preds[:,0].tolist()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=ts1, 
                    mode='lines',
                    name='Prediction'))
    fig.add_trace(go.Scatter(x=t, y=ts, 
                    mode='lines',
                    name='Original'))
    st.plotly_chart(fig)
 
    



st.title('Crypto')

#crypto_name = st.text_input('Crypto Name')

cyptos = ['BTC', 'ETH', 'ADA', 'BNB', 'MATIC', 'DOGE', 'LTC', 
            'SNX', 'LINK', 'VET', 'FIL', 'EOS', 'HBAR', 'CHZ', 'ENJ']
crypto_name = st.selectbox('Which algorithm?', cyptos)

today = date.today()
 

start_date = datetime.datetime(2015, 1, 1)
end_date = datetime.datetime(today.year,today.month,today.day)
    
yf_data = yf.download(crypto_name + '-USD', 
                        start=start_date, 
                            end=end_date, 
                                progress=False)


#historical_data = yf_data[['Adj Close','Open', 'High', 'Low', 'Close', 'Volume']].round(2)
historical_data = yf_data[['Adj Close', 'Open', 'Close']].round(4)
historical_data.reset_index(inplace = True)

time_step = 5
for_periods = 2



if st.checkbox('Show dataframe'):
    st.write(historical_data.tail(10))


st.subheader('Line plot')
col1 = st.selectbox('Which feature on y1?', historical_data.columns[1:4])
col2 = st.selectbox('Which feature on y2?', historical_data.columns[1:4])
#fig = px.line(historical_data, col1, col2)
fig = go.Figure()
fig.add_trace(go.Scatter(x=historical_data['Date'].values, y=historical_data[col1].values, 
                    mode='lines',
                    name=col1))
fig.add_trace(go.Scatter(x=historical_data['Date'].values, y=historical_data[col2].values*1.1, 
                    mode='lines',
                    name=col2))
st.plotly_chart(fig)


st.subheader('Model Regression')
reg_date = datetime.datetime(today.year-1,today.month,today.day)
end_date = datetime.datetime(today.year,today.month,today.day)
    
yf_data_reg = yf.download(crypto_name + '-USD', 
                        start=reg_date, 
                            end=end_date, 
                                progress=False)


#historical_data = yf_data[['Adj Close','Open', 'High', 'Low', 'Close', 'Volume']].round(2)
historical_data_reg = yf_data_reg[['Adj Close', 'Open', 'Close']].round(4)
historical_data_reg.reset_index(inplace = True)
rp = sns.regplot(x=historical_data_reg.index, y='Adj Close', data=historical_data_reg, ci=None, color='r')
y_rp = rp.get_lines()[0].get_ydata()
x_rp = rp.get_lines()[0].get_xdata()
fig = go.Figure()
fig.add_trace(go.Scatter(x=historical_data_reg.index, y=historical_data_reg['Adj Close'].values, 
                    mode='lines',
                    name='Adj Close'))
fig.add_trace(go.Scatter(x=x_rp, y=y_rp, 
                    mode='lines', name='Regression Fit'))
fig.add_trace(go.Scatter(x=x_rp, y=y_rp+ np.std(y_rp), 
                    mode='lines', name='Sell Line'))
fig.add_trace(go.Scatter(x=x_rp, y=y_rp - np.std(y_rp), 
                    mode='lines', name='Buy Line'))
st.plotly_chart(fig)




if st.checkbox('Prediction'):
    X_train, y_train, X_test, sc = ts_train_test_normalize(historical_data, time_step, for_periods)
    my_LSTM_model, LSTM_prediction = LSTM_model(X_train, y_train, X_test, sc, for_periods)
    actual_pred_plot(LSTM_prediction, historical_data) 






# st.subheader('Machine Learning models')
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn.svm import SVC

# features= historical_data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
# labels = historical_data['variety'].values
# X_train,X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)
# alg = ['Decision Tree', 'Support Vector Machine']
# classifier = st.selectbox('Which algorithm?', alg)
# if classifier=='Decision Tree':
#     dtc = DecisionTreeClassifier()
#     dtc.fit(X_train, y_train)
#     acc = dtc.score(X_test, y_test)
#     st.write('Accuracy: ', acc)
#     pred_dtc = dtc.predict(X_test)
#     cm_dtc=confusion_matrix(y_test,pred_dtc)
#     st.write('Confusion matrix: ', cm_dtc)
# elif classifier == 'Support Vector Machine':
#     svm=SVC()
#     svm.fit(X_train, y_train)
#     acc = svm.score(X_test, y_test)
#     st.write('Accuracy: ', acc)
#     pred_svm = svm.predict(X_test)
#     cm=confusion_matrix(y_test,pred_svm)
#     st.write('Confusion matrix: ', cm)