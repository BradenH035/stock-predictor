import pandas as pd
import os
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler

dir_path = './data'
data = []
csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
for file in csv_files:
    path = os.path.join(dir_path, file)
    temp = pd.read_csv(path)
    data.append({'name': os.path.splitext(file)[0], 'data': temp})

df = pd.DataFrame(data)

# Organize/Filter the data
new_order = ['Date', 'Open', 'Close', 'Volume']
def parse_dollar(dollar):
    return Decimal(dollar.replace('$', ''))


for i in range(len(df)):
    df['data'][i]['Date'] = pd.to_datetime(df['data'][i]['Date'], format='%m/%d/%Y')
    df['data'][i] = df['data'][i].drop(columns=['High', 'Low'])
    df['data'][i] = df['data'][i].rename(columns={'Close/Last': 'Close'})
    # parse dollar values (str --> decimal)
    open_values = df['data'][i]['Open']
    close_values = df['data'][i]['Close']
    new_open_values = [parse_dollar(value) for value in open_values]
    new_close_values = [parse_dollar(value) for value in close_values]
    df['data'][i]['Open'] = new_open_values
    df['data'][i]['Close'] = new_close_values
    # reorder
    df['data'][i] = df['data'][i][new_order]
    df['data'][i].index = df['data'][i]['Date']


# create plots for each of the stocks
for i in range(len(df)):
    plt.plot(df['data'][i]['Date'], df['data'][i]['Close'], label=df['name'][i])
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Closing Price for ' + df['name'][i])
    plt.savefig(f'./plots/{df["name"][i]}.png')
    plt.close()
    
# Train a model to predict the stock price based on the closing prices from the last 7 days
# Predict the closing price for the next day
# Use the model to predict the closing price for the next day
# Compare the predicted price to the actual price
# Calculate the difference between the predicted price and the actual price
# Calculate the average difference between the predicted price and the actual price

new_df = pd.DataFrame(index = range(0, len(df)), columns = ['Date', 'Close'])
for i in range(len(df)):
    new_df['Date'][i] = df['data'][i]['Date']
    new_df['Close'][i] = df['data'][i]['Close']

final_df = new_df.values
new_df.index = new_df.Date
new_df.drop('Date', axis=1, inplace=True)

train_data = final_df[0:1800, :]
valid_data = final_df[1800:, :]

print(final_df)
# issue, final_df is only dates
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(final_df['Close'])

x_train, y_train = [], []

# 60 days ~= 2 months
for i in range(60,len(train_data)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

inputs = new_df[len(new_df) - len(valid_data) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_closing_price = model.predict(X_test)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

model.save('stock_predictor.h5')

train_data = new_df[:1800]
valid_data = new_df[1800:]
valid_data['Predictions'] = predicted_closing_price
plt.plot(train_data['Close'])
plt.plot(valid_data[['Close','Predictions']])
plt.show()



