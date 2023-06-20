# Import any relevant packages or libraries
import pandas as pd
from scipy import stats

# Load dataset into dataframe
df = pd.read_csv('waze_dataset.csv')

# 1. Create `map_dictionary`
map_dictionary = {'Android': 2, 'iPhone': 1}

# 2. Create new `device_type` column
df['device_type'] = df['device']

# 3. Map the new column to the dictionary
df['device_type'] = df['device_type'].map(map_dictionary)
df['device_type'].head()
df.groupby('device_type')['drives'].mean()

# 1. Isolate the `drives` column for iPhone users.
iPhone = df[df['device_type'] == 1]['drives']

# 2. Isolate the `drives` column for Android users.
Android = df[df['device_type'] == 2]['drives']

# 3. Perform the t-test
t_stat, p_value = stats.ttest_ind(a=iPhone, b=Android, equal_var=False)

# Print the t-statistic and p-value
print("T-statistic:", t_stat)
print("P-value:", p_value)
