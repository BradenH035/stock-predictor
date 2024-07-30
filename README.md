This is a personal project based on starter code from https://data-flair.training/blogs/stock-price-prediction-machine-learning-project-in-python/

The goal was to create an application that would train models to predict the closing value of each stock. I used keras and scikit-learn within the program.
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

Checkout the outfile.txt file to see results from most recent test

Overall, the program works well and is relatively efficient given the amount of data passed through. Currently playing around with training variables to improve accuracy while ensuring that the data is not overtrained.

The next step will be to test this against real time data and see how it holds up. 
My goal with this project is to gain a better understanding of the stock market and machine learning. I have always been impressed with the people
that are proficient in the stock market, as it seems so complex and, at times, random. I learned that there is a cause behind each fluctuation and that it can be predict, to some degree.
