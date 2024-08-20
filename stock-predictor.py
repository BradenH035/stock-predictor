import pandas as pd
import os
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

# Load data
dir_path = './data'
data = []
csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
for file in csv_files:
    path = os.path.join(dir_path, file)
    temp = pd.read_csv(path)
    reversed = temp.iloc[::-1].reset_index(drop=True) # csv is stored date-wise from newest to oldest, so reverse it
    data.append({'name': os.path.splitext(file)[0], 'data': reversed})

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

# Create plots for each of the stocks
for i in range(len(df)):
    plt.plot(df['data'][i]['Date'], df['data'][i]['Close'], label=df['name'][i])
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Closing Price for ' + df['name'][i])
    plt.savefig(f'./plots/{df["name"][i]}.png')
    plt.close()

# Function to create training and testing datasets for multi-step predictions
def create_dataset(stock_data, time_step=60, prediction_horizon=1):
    X, y = [], []
    for i in range(time_step, len(stock_data) - prediction_horizon + 1):
        X.append(stock_data[i-time_step:i, 0])
        y.append(stock_data[i + prediction_horizon - 1, 0])
    return np.array(X), np.array(y)

def apply_weighting(data, decay_rate=0.995):
    """
    Apply exponential decay to the input data, giving less weight to older data points.
    UNUSED AS OF NOW
    """
    weights = [Decimal(decay_rate)**Decimal(i) for i in range(len(data))]
    weights = np.array(weights[::-1]).reshape(-1, 1)
    
    weighted_data = np.array([data[i] * weights[i][0] for i in range(len(data))])
    return weighted_data

# Train models for different horizons
horizons = {'1_day': 1, '1_week': 7, '1_month': 30}
models = {h: {} for h in horizons.keys()}

for h_name, h_days in horizons.items():
    for i in range(len(df)):
        stock_name = df['name'][i]
        stock_data = df['data'][i]['Close'].values
        stock_data = stock_data.reshape(-1, 1)

        # Apply weighting to the stock data
        weighted_stock_data = apply_weighting(stock_data)

        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_data)
        
        # Split data into training and validation sets
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        valid_data = scaled_data[train_size:]
        
        # Create datasets for the specified prediction horizon
        x_train, y_train = create_dataset(train_data, time_step=90, prediction_horizon=h_days)
        x_valid, y_valid = create_dataset(valid_data, time_step=90, prediction_horizon=h_days)
        
        # Reshape data to 3D (for LSTM input)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], 1)
        
        # Build LSTM
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train/save model
        model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_valid, y_valid))
        models[h_name][stock_name] = model


# Create the directory if it does not exist
os.makedirs('./models', exist_ok=True)

# Save the models dictionary
with open('./models/stock_models.pkl', 'wb') as f:
    pickle.dump(models, f)

print("Models trained and saved successfully!")

# Load the models dictionary
with open('./models/stock_models.pkl', 'rb') as f:
    models = pickle.load(f)

predictions = {h: {} for h in horizons.keys()}
for h_name, h_days in horizons.items():
    for stock_name, model in models[h_name].items():
        matching_rows = df[df['name'] == stock_name]
        close_values = []

        for index, row in matching_rows.iterrows():
            data_df = row['data'] 
            if 'Close' in data_df.columns:
                close_values.extend(data_df['Close'].values)

        stock_data = np.array(close_values)
        stock_data = stock_data.reshape(-1, 1)
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_data)
        
        # Get the last 90 day closing price values and scale the data
        last_90_days = scaled_data[-90:]
        
        X_test = []
        X_test.append(last_90_days)
        
        # Reformat and reshape data
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        pred_price = model.predict(X_test)
        
        # Undo scaling and add prediction to dictionary
        pred_price = scaler.inverse_transform(pred_price)
        predictions[h_name][stock_name] = pred_price[0][0]

# Output predictions for each horizon
for h_name in horizons.keys():
    print(f"Predictions for {h_name}:")
    for stock_name, pred_price in predictions[h_name].items():
        print(f"{stock_name}: Predicted {h_name} price: {pred_price}")
    print("\n")
   
# Get the actual price for a stock
for stock_name, pred_price in predictions.items():
    matching_rows = df[df['name'] == stock_name]

    close_values = []

    for index, row in matching_rows.iterrows():
        data_df = row['data']
        if 'Close' in data_df.columns:
            close_values.extend(data_df['Close'].values)

    actual_price = close_values[-1]
    prev_day_price = close_values[-2]
    last_week_price = close_values[-7]
    last_month_price = close_values[-30]
    print(f"Actual price for {stock_name} is: {actual_price}")
    print(f"Predicted price for {stock_name} is: {pred_price}")
    print(f"Error: {actual_price - Decimal(float(np.float32(pred_price)))}")

    print(f"Change in price from yesterday: {actual_price - prev_day_price}")
    print(f"Change in price from last week: {actual_price - last_week_price}")
    print(f"Change in price from last month: {actual_price - last_month_price}")
    major_change = Decimal(0.05) * actual_price

    points = 0 # points for the stock (based on price changes over different time periods)
    # if points > 0, then the stock is a good buy/hold
    # if points < 0, then the stock is a good sell

    if actual_price > prev_day_price:
        points += 1
        if actual_price - prev_day_price > major_change:
            print("Major price increase")
            points += 1
        if prev_day_price > last_week_price:
            print(f"Price has been increasing over the past week")
            points += 1
            if prev_day_price - last_week_price > major_change:
                print("Major price increase over the past week")
                points += 1
        else:
            print(f"Price has been fluctuating over the past week")
            points -= 1

        if actual_price > last_month_price:
            print(f"Price has been increasing over the past month")
            points += 1
            if actual_price - last_month_price > major_change:
                print("Major price increase over the past month")
                points += 1
        else: 
            print(f"Price has been fluctuating over the past month")
            points -= 1

    else:
        points -= 1
        if prev_day_price - actual_price > major_change:
            print("Major price decrease")
            points -= 1
        if prev_day_price < last_week_price:
            print(f"Price has been decreasing over the past week")
            points -= 1
            if last_week_price - prev_day_price > major_change:
                print("Major price decrease over the past week")
                points -= 1
        else:
            print(f"Price has been fluctuating over the past week")
            points += 1

        if actual_price < last_month_price:
            print(f"Price has been decreasing over the past month")
            points -= 1
            if last_month_price - actual_price > major_change:
                print("Major price decrease over the past month")
                points -= 1
        else:
            print(f"Price has been fluctuating over the past month")
            points += 1
          
    print("That concludes information for: " + stock_name)
    print(f"Points for {stock_name}: {points}")
    print("\n")


