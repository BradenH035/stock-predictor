import pandas as pd
import os
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt

dir_path = './data'
data = []
csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
for file in csv_files:
    print(file)
    path = os.path.join(dir_path, file)
    temp = pd.read_csv(path)
    data.append({'name': os.path.splitext(file)[0], 'data': temp})

df = pd.DataFrame(data)
print(df)

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
    


print(df['data'][0].head())
