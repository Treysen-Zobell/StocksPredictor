
"""
Inputs:
"""

from numpy import genfromtxt
from keras.models import Sequential
from keras.layers import Dense
import yfinance as yf


def download_stock_history(symbol, start='2020-01-01', end='2021-01-01'):
    data = yf.download(symbol, start=start, end=end)
    return data


def download_stock_current(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period='1d')
    return data['Close'][0]


def main():
    stock_history = download_stock_history('TSLA')
    stock_history.to_csv('tsla.csv')

    # load dataset
    dataset = genfromtxt('tsla.csv', delimiter=',')
    x = dataset[1:, 1: 5]
    y = dataset[1:, 6]

    # define keras model
    model = Sequential()
    model.add(Dense(12, input_dim=7, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit model to dataset
    model.fit(x, y, epochs=150, batch_size=10)

    # evaluate model
    _, accuracy = model.evaluate(x, y)
    print(f'Accuracy: {accuracy*100}')

    prediction = model.predict(x)
    for i in range(10):
        print(f'{x[i].tolist()} -> {prediction[i]} (expected {y[i]})')


if __name__ == '__main__':
    main()
