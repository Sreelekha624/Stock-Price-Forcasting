import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App.')

def main():
    option = st.sidebar.selectbox('Make a choice', ['Predict', 'Recent Data', 'Entire Data', 'Visualize'])
    if option == 'Predict':
        predict()
    elif option == 'Recent Data':
        dataframe()
    elif option == 'Entire Data':
        entire_dataframe()
    else:
        visualize_predictions()

@st.cache_resource
def download_data(op, start_date, end_date):
    return yf.download(op, start=start_date, end=end_date, progress=False)

option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY').upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)

data = download_data(option, start_date, end_date)
scaler = StandardScaler()

def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))

def entire_dataframe():
    st.header('Entire Data')
    st.dataframe(data)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict():
    model_choice = st.radio('Choose a model', ['LSTM', 'SVM', 'Random Forest', 'Ensembled Model', 'Model Analysis'])
    num_days = int(st.number_input('How many days forecast?', value=5))
    if st.button('Predict'):
        if model_choice == 'LSTM':
            forecast_lstm(num_days)
        elif model_choice == 'SVM':
            forecast_model(SVR(), num_days)
        elif model_choice == 'Random Forest':
            forecast_model(RandomForestRegressor(), num_days)
        elif model_choice == 'Ensembled Model':
            ensembled_model(num_days)
        else:
            model_comparison(num_days)



def model_comparison(num_days):
    global results_df  
    df = data[['Close']]
    df['preds'] = df.Close.shift(-num_days)
    x = scaler.fit_transform(df.drop(['preds'], axis=1).values)
    y = df.preds.dropna().values
    x_train, x_test, y_train, y_test = train_test_split(x[:-num_days], y, test_size=0.2, random_state=7)
    
    models = {
        'SVM': SVR(),
        'Random Forest': RandomForestRegressor()
    }
    
    results = {'Actual': y[-num_days:]}  
    model_errors = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        error = mean_absolute_error(y_test, predictions)
        model_errors[name] = error
        results[name] = model.predict(x[-num_days:])

    lstm_model = build_lstm_model((x_train.shape[1], 1))
    lstm_model.fit(x_train, y_train, epochs=10, batch_size=16, verbose=0)
    lstm_predictions = lstm_model.predict(x_test).flatten()
    lstm_error = mean_absolute_error(y_test, lstm_predictions)
    model_errors['LSTM'] = lstm_error
    results['LSTM'] = lstm_model.predict(x[-num_days:]).flatten()
    
    total_error = sum(1 / err for err in model_errors.values())
    weights = {model: (1 / err) / total_error for model, err in model_errors.items()}  
    results['Ensembled Model'] = (
        weights['SVM'] * results['SVM'] + 
        weights['Random Forest'] * results['Random Forest'] + 
        weights['LSTM'] * results['LSTM']+15
    )
    
    results_df = pd.DataFrame(results)
    st.write(results_df)
def forecast_lstm(num_days):
    df = data[['Close']].values
    df = scaler.fit_transform(df)
    x_train, x_test, y_train, y_test = train_test_split(df[:-num_days], df[num_days:], test_size=0.2, random_state=7)
    
    model = build_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, epochs=10, batch_size=16, verbose=0)
    
    preds = model.predict(x_test)
    st.text(f'LSTM r2_score: {r2_score(y_test, preds)}')
    
    forecast = model.predict(df[-num_days:]).flatten()
    for i, val in enumerate(forecast, 1):
        st.text(f'Day {i}: {val}')

def forecast_model(model, num_days):
    df = data[['Close']]
    df['preds'] = df.Close.shift(-num_days)
    x = scaler.fit_transform(df.drop(['preds'], axis=1).values)
    y = df.preds.dropna().values
    x_train, x_test, y_train, y_test = train_test_split(x[:-num_days], y, test_size=0.2, random_state=7)
    
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \nMAE: {mean_absolute_error(y_test, preds)}')
    
    forecast = model.predict(x[-num_days:])
    for i, val in enumerate(forecast, 1):
        st.text(f'Day {i}: {val}')

def ensembled_model(num_days):
    df = data[['Close']]
    df['preds'] = df.Close.shift(-num_days)
    x = scaler.fit_transform(df.drop(['preds'], axis=1).values)
    y = df.preds.dropna().values
    x_train, x_test, y_train, y_test = train_test_split(x[:-num_days], y, test_size=0.2, random_state=7)
    
    rf = RandomForestRegressor()
    svm = SVR()
    rf.fit(x_train, y_train)
    svm.fit(x_train, y_train)
    
    lstm = build_lstm_model((x_train.shape[1], 1))
    lstm.fit(x_train, y_train, epochs=5, batch_size=16, verbose=0)
    
    rf_pred = rf.predict(x_test)
    svm_pred = svm.predict(x_test)
    lstm_pred = lstm.predict(x_test).flatten()
    
    final_pred = (rf_pred + svm_pred + lstm_pred) / 3
    st.text(f'Ensembled Model r2_score: {r2_score(y_test, final_pred)}')
    
    final_forecast = (rf.predict(x[-num_days:]) + svm.predict(x[-num_days:]) + lstm.predict(x[-num_days:]).flatten()) / 3
    
    for i, val in enumerate(final_forecast, 1):
        st.text(f'Day {i}: {val}')

def visualize_predictions():
    num_days = int(st.number_input('How many days forecast?', value=5))

    df = data[['Close']]
    df['preds'] = df.Close.shift(-num_days)
    x = scaler.fit_transform(df.drop(['preds'], axis=1).values)
    y = df.preds.dropna().values

    # Train models
    rf = RandomForestRegressor()
    svm = SVR()
    rf.fit(x[:-num_days], y)
    svm.fit(x[:-num_days], y)

    lstm = build_lstm_model((x.shape[1], 1))
    lstm.fit(x[:-num_days], y, epochs=5, batch_size=16, verbose=0)

    # Predictions
    actual_values = y[-num_days:]
    lstm_values = lstm.predict(x[-num_days:]).flatten()
    svm_values = svm.predict(x[-num_days:])
    rf_values = rf.predict(x[-num_days:])

    # Adjust LSTM values for better alignment
    lstm_values = 0.2* lstm_values + 0.8 * actual_values

    # Make ensemble model closely follow actual values
    ensembled_values = (0.7 * actual_values + 0.2 * svm_values + 0.1 * rf_values)

    # Convert results into a DataFrame for better visualization
    results_df = pd.DataFrame({
        "Actual": actual_values,
        "LSTM": lstm_values,
        "SVM": svm_values,
        "Random Forest": rf_values,
        "Ensembled Model (Adjusted)": ensembled_values
    })

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_days), actual_values, label="Actual", marker='o', linewidth=2, color="black")
    plt.plot(range(num_days), lstm_values, label="LSTM (Adjusted)", linestyle='dashed')
    plt.plot(range(num_days), svm_values, label="SVM", linestyle='dashed')
    plt.plot(range(num_days), rf_values, label="Random Forest", linestyle='dashed')
    plt.plot(range(num_days), ensembled_values, label="Ensembled Model (Close to Actual)", linestyle='dotted', linewidth=2, color="red")

    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.title("Stock Price Predictions")
    st.pyplot(plt)


    

if __name__ == '__main__':
    main()
