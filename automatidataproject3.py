import pandas as pd
from scipy import stats

# Load the taxi data into a dataframe
taxi_data = pd.read_csv("2017_Yellow_Taxi_Trip_Data.csv", index_col=0)

# Perform descriptive statistics for the data
taxi_data.describe(include='all')

# Calculate the mean total amount for each payment type
mean_credit_card = taxi_data[taxi_data['payment_type'] == 1]['total_amount'].mean()
mean_cash = taxi_data[taxi_data['payment_type'] == 2]['total_amount'].mean()

# Print the mean total amounts for each payment type
print("Mean total amount for credit card payments:", mean_credit_card)
print("Mean total amount for cash payments:", mean_cash)

# Select the total amounts for credit card and cash payments
credit_card = taxi_data[taxi_data['payment_type'] == 1]['total_amount']
cash = taxi_data[taxi_data['payment_type'] == 2]['total_amount']

# Perform a two-sample t-test to compare the mean total amounts for credit card and cash payments
t_stat, p_value = stats.ttest_ind(a=credit_card, b=cash, equal_var=False)

# Print the t-statistic and p-value
print("T-statistic:", t_stat)
print("P-value:", p_value)
