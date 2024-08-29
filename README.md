This is a personal project inspired by the starter code from https://data-flair.training/blogs/stock-price-prediction-machine-learning-project-in-python/

My goal with this project is to gain a better understanding of the stock market and machine learning by creating an application that will train models to predict the closing value of each stock. I used keras and scikit-learn in the program.
Data for each stock was gathered from Nasdaq.com (https://www.nasdaq.com/market-activity/quotes/historical)

Simplified Breakdown:
1) Read each csv file and store in separate dataframe
2) Filter data
3) Create a plot of the closing value for each stock over the last decade (provides a visual for what values to expect)
4) Break data into training/testing datasets
5) Train model for each of the stocks on their designated training dataset
   - Comments within code provide more detailed explaination of this process
7) Save models
8) Load models
9) Predict closing values & compare to actual data

Checkout the "trainingVariations" folder to see results from recent testing!

Overall, the program works well and is relatively efficient given the amount of data passed through. Currently experimenting with training variables to improve accuracy while ensuring that the data is not overtrained. Another challenge comes from aspects that are hard to quantify with numbers, such as the release of an exciting new product that causes stock prices to jump, or a major controversey that negatively impacts the stock price. I learned that, while it is possible to predict the stock prices within a high degree of accuracy solely using raw data, numbers alone are not enough to understand each change in price.

The next step will be to test this on data in real time and see how it holds up. 
