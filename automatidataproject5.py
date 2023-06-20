import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance

pd.set_option('display.max_columns', None)

# Load dataset into dataframe
df0 = pd.read_csv('2017_Yellow_Taxi_Trip_Data.csv')
nyc_preds_means = pd.read_csv('nyc_preds_means.csv')
df1 = df0.merge(nyc_preds_means, left_index=True, right_index=True)

# Subset the data to isolate only customers who paid by credit card
df1 = df1[df1['payment_type'] == 1]

# Create tip % col
df1['tip_percent'] = df1['tip_amount'] / (df1['total_amount'] - df1['tip_amount'])

# Create 'generous' col (target)
df1['generous'] = (df1['tip_percent'] >= 0.2).astype(int)

# Convert pickup and dropoff cols to datetime
df1['tpep_pickup_datetime'] = pd.to_datetime(df1['tpep_pickup_datetime'], format='%m/%d/%Y %H:%M')

# Create time-based columns
df1['am_rush'] = ((df1['tpep_pickup_datetime'].dt.hour >= 6) & (df1['tpep_pickup_datetime'].dt.hour < 10)).astype(int)
df1['daytime'] = ((df1['tpep_pickup_datetime'].dt.hour >= 10) & (df1['tpep_pickup_datetime'].dt.hour < 16)).astype(int)
df1['pm_rush'] = ((df1['tpep_pickup_datetime'].dt.hour >= 16) & (df1['tpep_pickup_datetime'].dt.hour < 20)).astype(int)
df1['nighttime'] = (((df1['tpep_pickup_datetime'].dt.hour >= 20) & (df1['tpep_pickup_datetime'].dt.hour < 24)) |
                    ((df1['tpep_pickup_datetime'].dt.hour >= 0) & (df1['tpep_pickup_datetime'].dt.hour < 6))).astype(int)

# Create 'month' col
df1['month'] = df1['tpep_pickup_datetime'].dt.strftime('%b').str.lower()

# Drop unnecessary columns
drop_cols = ['Unnamed: 0', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'payment_type', 'trip_distance',
             'store_and_fwd_flag', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
             'improvement_surcharge', 'total_amount', 'tip_percent']
df1 = df1.drop(drop_cols, axis=1)

# Convert categoricals to binary
df2 = pd.get_dummies(df1, drop_first=True)

# Get class balance of 'generous' col
df2['generous'].value_counts()

# Split into train and test sets
X = df2.drop('generous', axis=1)
y = df2['generous']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Instantiate the random forest classifier
rf = RandomForestClassifier(random_state=42)

# Define a dictionary of hyperparameters to tune
cv_params = {'max_depth': [None],
             'max_features': [1.0],
             'max_samples': [0.7],
             'min_samples_leaf': [1],
             'min_samples_split': [2],
             'n_estimators': [300]
             }

# Instantiate the GridSearchCV object
rf1 = GridSearchCV(rf, cv_params, scoring='f1', cv=4, refit='f1')

# Measure execution time
start_time = time.time()

# Fit the model
rf1.fit(X_train, y_train)

# Print execution time
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

# Print best score and best parameters
print("Best F1 score:", rf1.best_score_)
print("Best parameters:", rf1.best_params_)

# Get scores on test data
preds = rf1.best_estimator_.predict(X_test)

# Define a path to the folder where you want to save the model
path = r'C:\Users\short\OneDrive\Desktop\portfolio\automatidataproject\\'

# Save the model
with open(path + 'taxi_rf_cv1.pickle', 'wb') as to_write:
    pickle.dump(rf1, to_write)

# Load the saved model
with open(path + 'taxi_rf_cv1.pickle', 'rb') as to_read:
    rf_cv1 = pickle.load(to_read)

# Define a function to calculate evaluation metrics
def get_test_scores(model_name, preds, y_test_data):
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy]
                          })

    return table

results = pd.DataFrame()
# Calculate test scores for the random forest model
rf_test_scores = get_test_scores('RF test', preds, y_test)
results = pd.concat([results, rf_test_scores], axis=0)
results

# Instantiate the XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state=0)

# Define a dictionary of hyperparameters to tune
cv_params = {'learning_rate': [0.1],
             'max_depth': [8],
             'min_child_weight': [2],
             'n_estimators': [500]
             }

# Instantiate the GridSearchCV object
xgb1 = GridSearchCV(xgb, cv_params, scoring='f1', cv=4, refit='f1')

# Measure execution time
start_time = time.time()

# Fit the model
xgb1.fit(X_train, y_train)

# Print execution time
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

# Print best score and best parameters
print("Best F1 score:", xgb1.best_score_)
print("Best parameters:", xgb1.best_params_)

# Get scores on test data
preds = xgb1.best_estimator_.predict(X_test)

# Save the model
with open(path + 'taxi_xgb_cv1.pickle', 'wb') as to_write:
    pickle.dump(xgb1, to_write)

# Load the saved model
with open(path + 'taxi_xgb_cv1.pickle', 'rb') as to_read:
    xgb_cv1 = pickle.load(to_read)

# Calculate test scores for the XGBoost model
xgb_test_scores = get_test_scores('XGBoost test', preds, y_test)
results = pd.concat([results, xgb_test_scores], axis=0)
results

# 1. Instantiate the XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state=0)

# 2. Create a dictionary of hyperparameters to tune
# Note that this example only contains 1 value for each parameter for simplicity,
# but you should assign a dictionary with ranges of values
cv_params = {'learning_rate': [0.1],
             'max_depth': [8],
             'min_child_weight': [2],
             'n_estimators': [500]
             }

# 3. Define a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1'}

# 4. Instantiate the GridSearchCV object
xgb1 = GridSearchCV(xgb, cv_params, scoring=scoring, cv=4, refit='f1')

# Measure execution time
start_time = time.time()

# Code block to measure

# Measure execution time
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

xgb1.fit(X_train, y_train)

# Examine best score
xgb1.best_score_

# Examine best parameters
xgb1.best_params_

def make_results(model_name, model, scoring_metric):
    cv_results = pd.DataFrame(model.cv_results_)
    results = cv_results[['param_learning_rate', 'param_max_depth', 'param_min_child_weight', 'param_n_estimators',
                          'mean_test_' + scoring_metric, 'rank_test_' + scoring_metric]]
    results.columns = ['learning_rate', 'max_depth', 'min_child_weight', 'n_estimators', 'mean_' + scoring_metric,
                       'rank_' + scoring_metric]
    results.loc[:, 'model'] = model_name
    return results


# Call 'make_results()' on the GridSearch object
xgb1_cv_results = make_results('XGB CV', xgb1, 'f1')
results = pd.concat([results, xgb1_cv_results], axis=0)
results

# Get scores on test data
preds = xgb1.best_estimator_.predict(X_test)

# Get scores on test data
xgb_test_scores = get_test_scores('XGB test', preds, y_test)
results = pd.concat([results, xgb_test_scores], axis=0)
results

# Generate array of values for confusion matrix
cm = confusion_matrix(y_test, preds, labels=xgb1.classes_)

# Plot confusion matrix
labels = xgb1.classes_
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

plot_importance(xgb1.best_estimator_, max_num_features=10)
plt.show()
