import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
import time

# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")

# Display first few rows
data.head()

# Get number of rows and columns
data.shape

# Get basic information
data.info()

# Generate basic descriptive stats
data.describe()

# Check for missing values
data.isna().sum()

# Drop rows with missing values
data = data.dropna(axis=0)

# Check for duplicates
data.duplicated().sum()

# Check class balance
data["claim_status"].value_counts(normalize=True)

# Create `text_length` column
data['text_length'] = data['video_transcription_text'].str.len()
data.head()

# Visualize the distribution of `video_transcription_text` length for claims and opinions
sns.histplot(data=data, stat="count", multiple="dodge", x="text_length",
             kde=False, palette="pastel", hue="claim_status",
             element="bars", legend=True)
plt.xlabel("video_transcription_text length (number of characters)")
plt.ylabel("Count")
plt.title("Distribution of video_transcription_text length for claims and opinions")
plt.show()

# Preprocess data
X = data.copy()
X['claim_status'] = X['claim_status'].replace({'opinion': 0, 'claim': 1})
X = pd.get_dummies(X, columns=['verified_status', 'author_ban_status'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.drop('claim_status', axis=1), X['claim_status'], test_size=0.2, random_state=0)

# Set up a `CountVectorizer` object
count_vec = CountVectorizer(ngram_range=(2, 3), max_features=15, stop_words='english')

# Extract numerical features from `video_transcription_text` in the training set
count_data = count_vec.fit_transform(X_train['video_transcription_text']).toarray()
count_df = pd.DataFrame(data=count_data, columns=count_vec.get_feature_names_out())

# Concatenate `X_train` and `count_df` to form the final dataframe for training data (`X_train_final`)
X_train_final = pd.concat([X_train.drop(columns=['video_transcription_text']).reset_index(drop=True), count_df], axis=1)

# Extract numerical features from `video_transcription_text` in the testing set
validation_count_data = count_vec.transform(X_test['video_transcription_text']).toarray()
validation_count_df = pd.DataFrame(data=validation_count_data, columns=count_vec.get_feature_names_out())

# Concatenate `X_test` and `validation_count_df` to form the final dataframe for testing data (`X_test_final`)
X_test_final = pd.concat([X_test.drop(columns=['video_transcription_text']).reset_index(drop=True), validation_count_df], axis=1)

# Instantiate the random forest classifier
rf = RandomForestClassifier(random_state=0)

# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [5, 7, None],
             'max_features': [0.3, 0.6],
             'max_samples': [0.7],
             'min_samples_leaf': [1,2],
             'min_samples_split': [2,3],
             'n_estimators': [75,100,200],
             }

# Define a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1'}

# Instantiate the GridSearchCV object
rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=5, refit='recall')

start_time = time.time()
rf_cv.fit(X_train_final, y_train)

# Examine best recall score
rf_cv.best_score_

# Examine best parameters
rf_cv.best_params_

# Instantiate the XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state=0)

# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [4,8,12],
             'min_child_weight': [3, 5],
             'learning_rate': [0.01, 0.1],
             'n_estimators': [300, 500]
             }

# Instantiate the GridSearchCV object
xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=5, refit='recall')

xgb_cv.fit(X_train_final, y_train)
xgb_cv.best_score_
xgb_cv.best_params_

# Use the random forest "best estimator" model to get predictions on the validation set
y_pred_rf = rf_cv.best_estimator_.predict(X_test_final)

# Create a confusion matrix to visualize the results of the random forest classification model
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=None)

# Plot the confusion matrix for the random forest model
disp_rf.plot()
plt.title('Random Forest - Test Set')
plt.show()

# Create a classification report for the random forest model
target_labels = ['opinion', 'claim']
print("Random Forest - Classification Report")
print(classification_report(y_test, y_pred_rf, target_names=target_labels))

# Use the XGBoost "best estimator" model to get predictions on the validation set
y_pred_xgb = xgb_cv.best_estimator_.predict(X_test_final)

# Create a confusion matrix to visualize the results of the XGBoost classification model
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=None)

# Plot the confusion matrix for the XGBoost model
disp_xgb.plot()
plt.title('XGBoost - Test Set')
plt.show()

# Create a classification report for the XGBoost model
print("XGBoost - Classification Report")
print(classification_report(y_test, y_pred_xgb, target_names=target_labels))

# Plot feature importances for the random forest model
importances_rf = rf_cv.best_estimator_.feature_importances_
feature_importances_rf = pd.Series(importances_rf, index=X_test_final.columns)

plt.figure(figsize=(10, 6))
feature_importances_rf.sort_values().plot(kind='barh')
plt.title('Random Forest - Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Calculate and print the execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
