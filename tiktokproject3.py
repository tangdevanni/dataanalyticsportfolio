# Import packages for data manipulation
import pandas as pd
import numpy as np

# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import packages for statistical analysis/hypothesis testing
from scipy import stats

# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")

# Display first few rows
data.head()

# Generate a table of descriptive statistics about the data
data.describe()

# Check for missing values
data.isna().sum()

# Drop rows with missing values
data = data.dropna(axis=0)

# Display first few rows after handling missing values
data.head()

# Compute the mean `video_view_count` for each group in `verified_status`
mean_not_verified = data[data["verified_status"] == "not verified"]["video_view_count"].mean()
mean_verified = data[data["verified_status"] == "verified"]["video_view_count"].mean()

# Print the mean `video_view_count` for each group
print("Mean video_view_count for 'not verified':", mean_not_verified)
print("Mean video_view_count for 'verified':", mean_verified)

# Conduct a two-sample t-test to compare means
not_verified = data[data["verified_status"] == "not verified"]["video_view_count"]
verified = data[data["verified_status"] == "verified"]["video_view_count"]

t_stat, p_value = stats.ttest_ind(a=not_verified, b=verified, equal_var=False)

# Print the t-statistic and p-value
print("T-statistic:", t_stat)
print("P-value:", p_value)
