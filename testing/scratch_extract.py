import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel

import ipywidgets as widgets
from IPython.display import display, HTML
from collections import defaultdict

import google.generativeai as genai
from google.colab import userdata

# --- CELL ---

try:
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "ravisinghiitbhu/nfhs5",
        "Final.csv"
    )
    print("Dataset loaded successfully.")
    display(df.head())
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure the dataset 'ravisinghiitbhu/nfhs5' is available and accessible.")

# --- CELL ---

# Display basic information, descriptive statistics, and target variable distribution.
print("Dataset Information:")
df.info()

print("\nDataset Description:")
display(df.describe())

print("\nTarget Variable ('ChildAlive') Distribution:")
display(df['ChildAlive'].value_counts())

# --- CELL ---

# Separate target variable
X = df.drop('ChildAlive', axis=1)
y = df['ChildAlive']

# Convert categorical 'State' column into numerical representation using one-hot encoding
X = pd.get_dummies(X, columns=['State'], drop_first=False) # Keep all states for explicit mapping

# Store feature names after preprocessing but before feature selection
original_feature_columns = X.columns.tolist()

print(f"Features shape after one-hot encoding 'State': {X.shape}")

# --- CELL ---

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Original training set shape: {X_train.shape}, {y_train.shape}")
print(f"Original testing set shape: {X_test.shape}, {y_test.shape}")
print(f"Original training class distribution:\n{y_train.value_counts()}")
print(f"Original testing class distribution:\n{y_test.value_counts()}")

# --- CELL ---

# Train an initial Logistic Regression model on the full feature set to get coefficients
initial_model_for_selection = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
initial_model_for_selection.fit(X_train, y_train)

# Initialize SelectFromModel to select features based on coefficients
# We aim for ~30-40 features. Let's try to get around 30 initially.
# The threshold is set to a very small negative value and max_features is used for selection.
selector = SelectFromModel(initial_model_for_selection, prefit=True, max_features=30, threshold=-np.inf)

# Transform the training and testing data to select only the most important features
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Get the names of the selected features
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = X.columns[selected_feature_indices]

print(f"Number of features selected by SelectFromModel: {len(selected_feature_names)}")
print(f"Selected features: {list(selected_feature_names)}")

# Define model_expected_columns for the prediction function
model_expected_columns = selected_feature_names.tolist()

# Update X_train and X_test to be the selected feature DataFrames (useful for subsequent steps)
X_train = pd.DataFrame(X_train_selected, columns=selected_feature_names)
X_test = pd.DataFrame(X_test_selected, columns=selected_feature_names)

# --- CELL ---

# Initialize and train a Logistic Regression model with the selected features
selected_model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
selected_model.fit(X_train, y_train)
print(X_train.columns.tolist())

print("Model trained successfully on selected features.")

# --- CELL ---

import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(selected_model, f)

print("Model saved successfully!")

# --- CELL ---

import pickle

# Save expected feature columns
with open("model_expected_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Expected columns saved!")

# --- CELL ---

# Make predictions on the test set using the model with selected features
y_pred_selected = selected_model.predict(X_test)

# Generate and print the classification report
print("Classification Report using Selected Features:")
print(classification_report(y_test, y_pred_selected))

# --- CELL ---

# Get feature names from the selected training data
feature_names = X_train.columns

# Get the coefficients from the trained Logistic Regression model
coefficients = selected_model.coef_[0]

# Create a DataFrame to store feature names and their coefficients
feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Calculate the absolute value of coefficients for ranking significance
feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])

# Sort by absolute coefficient in descending order
feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=False)

# Display the most significant features
print("Most Significant Features (after selection):")
display(feature_importance.head(len(feature_importance)))

# --- CELL ---

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Ensure pandas is imported

# Check if feature_importance is defined, if not, re-calculate it
if 'feature_importance' not in globals():
    print("feature_importance not found, re-calculating from selected_model and X_train.")
    # Assuming selected_model and X_train are available from previous cells
    if 'selected_model' in globals() and 'X_train' in globals():
        feature_names = X_train.columns
        coefficients = selected_model.coef_[0]
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
        feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
        feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=False)
    else:
        print("Could not re-calculate feature_importance: selected_model or X_train not found. Please run relevant model training and feature selection cells.")

# Visualize the top N features, only if feature_importance is available
if 'feature_importance' in globals():
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Abs_Coefficient', y='Feature', data=feature_importance.head(15), palette='viridis')
    plt.title('Top 15 Most Significant Features (Logistic Regression)')
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Feature')
    plt.show()
else:
    print("Skipping feature importance plot due to missing 'feature_importance' DataFrame.")

# --- CELL ---

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the correlation matrix for the selected features
correlation_matrix = X_train.corr()

plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Selected Features', fontsize=16)
plt.show()

# --- CELL ---

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Re-create necessary variables (X_train, y_train, X_test, y_test) if not defined
# This assumes 'X', 'y' are defined from Data Cleaning & Preprocessing (cell c67ae3a2)
# and 'selected_feature_names' from Feature Selection (cell 65014b0c)
if 'X_train' not in locals() or 'y_train' not in locals() or 'X_test' not in locals() or 'y_test' not in locals() or 'df' not in globals():
    print("Re-creating X_train, y_train, X_test, y_test for Decision Tree Classifier...")
    # 1. Get original X, y (assuming df and original_feature_columns are available)
    if 'df' not in globals():
        print("DataFrame 'df' not found. Reloading dataset.")
        try:
            # Replicate data loading from cell 17b27236
            df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "ravisinghiitbhu/nfhs5", "Final.csv")
            print("Dataset reloaded successfully.")
        except Exception as e:
            print(f"Error reloading dataset in Decision Tree cell: {e}")
            # If data loading fails, we cannot proceed. Raise an error.
            raise

    # Assuming df is now available
    X_full = df.drop('ChildAlive', axis=1)
    y_full = df['ChildAlive']
    X_full = pd.get_dummies(X_full, columns=['State'], drop_first=False)

    # 2. Re-split to get X_train_original and X_test_original before selection
    X_train_original, X_test_original, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

    # 3. Re-apply feature selection to X_train_original and X_test_original
    initial_model_for_selection = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
    initial_model_for_selection.fit(X_train_original, y_train)
    selector = SelectFromModel(initial_model_for_selection, prefit=True, max_features=30, threshold=-np.inf)

    X_train = pd.DataFrame(selector.transform(X_train_original), columns=X_full.columns[selector.get_support()])
    X_test = pd.DataFrame(selector.transform(X_test_original), columns=X_full.columns[selector.get_support()])
    selected_feature_names = X_full.columns[selector.get_support()]
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


# Initialize and train a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

print("Decision Tree Classifier trained successfully.")

# Make predictions on the test set
y_pred_dt = dt_classifier.predict(X_test)

# Generate and print the classification report
print("\nClassification Report for Decision Tree Classifier:")
print(classification_report(y_test, y_pred_dt))

dt_report = classification_report(y_test, y_pred_dt, output_dict=True)

# --- CELL ---

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Re-create necessary variables (X_train, y_train, X_test, y_test) if not defined
# This assumes 'X', 'y' are defined from Data Cleaning & Preprocessing (cell c67ae3a2)
# and 'selected_feature_names' from Feature Selection (cell 65014b0c)
if 'X_train' not in locals() or 'y_train' not in locals() or 'X_test' not in locals() or 'y_test' not in locals() or 'df' not in globals():
    print("Re-creating X_train, y_train, X_test, y_test for Random Forest Classifier...")
    # 1. Get original X, y (assuming df and original_feature_columns are available)
    if 'df' not in globals():
        print("DataFrame 'df' not found. Reloading dataset.")
        try:
            # Replicate data loading from cell 17b27236
            df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "ravisinghiitbhu/nfhs5", "Final.csv")
            print("Dataset reloaded successfully.")
        except Exception as e:
            print(f"Error reloading dataset in Random Forest cell: {e}")
            # If data loading fails, we cannot proceed. Raise an error.
            raise

    # Assuming df is now available
    X_full = df.drop('ChildAlive', axis=1)
    y_full = df['ChildAlive']
    X_full = pd.get_dummies(X_full, columns=['State'], drop_first=False)

    # 2. Re-split to get X_train_original and X_test_original before selection
    X_train_original, X_test_original, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

    # 3. Re-apply feature selection to X_train_original and X_test_original
    initial_model_for_selection = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
    initial_model_for_selection.fit(X_train_original, y_train)
    selector = SelectFromModel(initial_model_for_selection, prefit=True, max_features=30, threshold=-np.inf)

    X_train = pd.DataFrame(selector.transform(X_train_original), columns=X_full.columns[selector.get_support()])
    X_test = pd.DataFrame(selector.transform(X_test_original), columns=X_full.columns[selector.get_support()])
    selected_feature_names = X_full.columns[selector.get_support()]
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Initialize and train a Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
rf_classifier.fit(X_train, y_train)

print("Random Forest Classifier trained successfully.")

# Make predictions on the test set
y_pred_rf = rf_classifier.predict(X_test)

# Generate and print the classification report
print("\nClassification Report for Random Forest Classifier:")
print(classification_report(y_test, y_pred_rf))

rf_report = classification_report(y_test, y_pred_rf, output_dict=True)

# --- CELL ---

import pandas as pd
from sklearn.metrics import classification_report

# Initialize reports with placeholder values
# These will be used if actual data or predictions are not available
default_report = {
    '0': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
    '1': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
    'accuracy': 0.0,
    'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
    'weighted avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
}
lr_report = default_report.copy()
dt_report = default_report.copy()
rf_report = default_report.copy()

# Check if y_test is available
if 'y_test' not in locals():
    print("y_test not found. Please ensure the data splitting cells are run.")
else:
    # If y_test is defined, try to get actual reports
    # Logistic Regression Report
    if 'y_pred_selected' in locals():
        lr_report = classification_report(y_test, y_pred_selected, output_dict=True)
    else:
        print("lr_report and y_pred_selected not found. Using default placeholder for Logistic Regression.")
        # Use the predefined lr_report (which is default_report.copy())

    # Decision Tree Report
    if 'y_pred_dt' in locals():
        dt_report = classification_report(y_test, y_pred_dt, output_dict=True)
    else:
        print("dt_report and y_pred_dt not found. Using default placeholder for Decision Tree.")
        # Use the predefined dt_report (which is default_report.copy())

    # Random Forest Report
    if 'y_pred_rf' in locals():
        rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
    else:
        print("rf_report and y_pred_rf not found. Using default placeholder for Random Forest.")
        # Use the predefined rf_report (which is default_report.copy())


# Now, comparison_data can be safely constructed as lr_report, dt_report, rf_report are always defined
comparison_data = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [lr_report['accuracy'], dt_report['accuracy'], rf_report['accuracy']],
    'Precision (Class 0)': [lr_report['0']['precision'], dt_report['0']['precision'], rf_report['0']['precision']],
    'Recall (Class 0)': [lr_report['0']['recall'], dt_report['0']['recall'], rf_report['0']['recall']],
    'F1-Score (Class 0)': [lr_report['0']['f1-score'], dt_report['0']['f1-score'], rf_report['0']['f1-score']]
}

comparison_df = pd.DataFrame(comparison_data)
print("\nModel Performance Comparison:")
display(comparison_df.round(3))

# --- CELL ---

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure comparison_df is available and properly populated
if 'comparison_df' in locals() and not comparison_df.empty:
    metrics_to_plot = ['Accuracy', 'Precision (Class 0)', 'Recall (Class 0)', 'F1-Score (Class 0)']

    # Melt the DataFrame for easier plotting with seaborn
    df_melted = comparison_df.melt(id_vars='Model', value_vars=metrics_to_plot, var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 7))
    sns.barplot(x='Metric', y='Score', hue='Model', data=df_melted, palette='muted')
    plt.title('Comparison of Model Performance Metrics', fontsize=16)
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.ylim(0, 1) # Metrics are typically between 0 and 1
    plt.legend(title='Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("Comparison DataFrame is not available or empty. Please ensure previous cells generating model reports and comparison_df are run.")

# --- CELL ---

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Define the parameter grid for Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

# Initialize GridSearchCV
grid_search_lr = GridSearchCV(
    LogisticRegression(random_state=42, solver='liblinear', max_iter=1000, class_weight='balanced'),
    param_grid_lr,
    cv=5,
    scoring='f1', # Focus on F1-score for the minority class
    n_jobs=-1,
    verbose=1
)

print("Starting GridSearchCV for Logistic Regression...")
grid_search_lr.fit(X_train, y_train)

print("Logistic Regression tuning complete.")
print(f"Best parameters for Logistic Regression: {grid_search_lr.best_params_}")
print(f"Best F1-score (minority class) for Logistic Regression: {grid_search_lr.best_score_:.4f}")

best_lr_model = grid_search_lr.best_estimator_
# Evaluate on test set
y_pred_lr_tuned = best_lr_model.predict(X_test)
lr_tuned_report = classification_report(y_test, y_pred_lr_tuned, output_dict=True)
print("\nClassification Report for Tuned Logistic Regression:")
print(classification_report(y_test, y_pred_lr_tuned))

# --- CELL ---

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Define the parameter grid for Decision Tree
param_grid_dt = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Initialize GridSearchCV
grid_search_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    param_grid_dt,
    cv=5,
    scoring='f1', # Focus on F1-score for the minority class
    n_jobs=-1,
    verbose=1
)

print("Starting GridSearchCV for Decision Tree Classifier...")
grid_search_dt.fit(X_train, y_train)

print("Decision Tree tuning complete.")
print(f"Best parameters for Decision Tree: {grid_search_dt.best_params_}")
print(f"Best F1-score (minority class) for Decision Tree: {grid_search_dt.best_score_:.4f}")

best_dt_model = grid_search_dt.best_estimator_
# Evaluate on test set
y_pred_dt_tuned = best_dt_model.predict(X_test)
dt_tuned_report = classification_report(y_test, y_pred_dt_tuned, output_dict=True)
print("\nClassification Report for Tuned Decision Tree:")
print(classification_report(y_test, y_pred_dt_tuned))

# --- CELL ---

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Define the parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', None]
}

# Initialize GridSearchCV
grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=3, # Using lower CV for faster computation
    scoring='f1', # Focus on F1-score for the minority class
    n_jobs=-1,
    verbose=1
)

print("Starting GridSearchCV for Random Forest Classifier...")
grid_search_rf.fit(X_train, y_train)

print("Random Forest tuning complete.")
print(f"Best parameters for Random Forest: {grid_search_rf.best_params_}")
print(f"Best F1-score (minority class) for Random Forest: {grid_search_rf.best_score_:.4f}")

best_rf_model = grid_search_rf.best_estimator_
# Evaluate on test set
y_pred_rf_tuned = best_rf_model.predict(X_test)
rf_tuned_report = classification_report(y_test, y_pred_rf_tuned, output_dict=True)
print("\nClassification Report for Tuned Random Forest:")
print(classification_report(y_test, y_pred_rf_tuned))

# --- CELL ---

import pandas as pd

# Ensure reports are available. If tuning wasn't run, use default placeholders.
if 'lr_tuned_report' not in locals(): lr_tuned_report = default_report.copy()
if 'dt_tuned_report' not in locals(): dt_tuned_report = default_report.copy()
if 'rf_tuned_report' not in locals(): rf_tuned_report = default_report.copy()

comparison_data_tuned = {
    'Model': ['Tuned Logistic Regression', 'Tuned Decision Tree', 'Tuned Random Forest'],
    'Accuracy': [lr_tuned_report['accuracy'], dt_tuned_report['accuracy'], rf_tuned_report['accuracy']],
    'Precision (Class 0)': [lr_tuned_report['0']['precision'], dt_tuned_report['0']['precision'], rf_tuned_report['0']['precision']],
    'Recall (Class 0)': [lr_tuned_report['0']['recall'], dt_tuned_report['0']['recall'], rf_tuned_report['0']['recall']],
    'F1-Score (Class 0)': [lr_tuned_report['0']['f1-score'], dt_tuned_report['0']['f1-score'], rf_tuned_report['0']['f1-score']]
}

comparison_df_tuned = pd.DataFrame(comparison_data_tuned)
print("\nTuned Model Performance Comparison:")
display(comparison_df_tuned.round(3))


# --- CELL ---

import matplotlib.pyplot as plt
import seaborn as sns

if 'comparison_df_tuned' in locals() and not comparison_df_tuned.empty:
    metrics_to_plot = ['Accuracy', 'Precision (Class 0)', 'Recall (Class 0)', 'F1-Score (Class 0)']

    df_melted_tuned = comparison_df_tuned.melt(id_vars='Model', value_vars=metrics_to_plot, var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 7))
    sns.barplot(x='Metric', y='Score', hue='Model', data=df_melted_tuned, palette='dark')
    plt.title('Comparison of Tuned Model Performance Metrics', fontsize=16)
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.ylim(0, 1)
    plt.legend(title='Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("Tuned Comparison DataFrame is not available or empty. Please ensure previous tuning cells are run.")

# --- CELL ---

# --- Mapping Technical Names to Layman Labels (Expanded) ---
# This dictionary maps technical feature names to human-friendly questions or labels for the UI.
feature_labels = {
    'Res_Age': "What is the mother's current age in years?",
    'Edu_level': 'What is the highest education level completed by the mother? (0: No education, 1: Primary, 2: Secondary, 3: Higher)',
    'Water_Source_Time': 'How many minutes does it take to fetch water? (Enter 0 if water is available on the premises)',
    'Toilet_Facility': 'Does the household have access to a toilet facility? (1: Yes, 0: No)',
    'House_electricity': 'Does the house have electricity? (1: Yes, 0: No)',
    'House_radio': 'Does the household own a radio? (1: Yes, 0: No)',
    'House_tv': 'Does the household own a television? (1: Yes, 0: No)',
    'House_bicycle': 'Does the household own a bicycle? (1: Yes, 0: No)',
    'House_motorcycle': 'Does the household own a motorcycle or scooter? (1: Yes, 0: No)',
    'House_car': 'Does the household own a car/truck? (1: Yes, 0: No)',
    'Household_members': 'What is the total number of people living in the household?',
    'Child_under5': 'What is the number of children under age 5 currently living in the household?',
    'House_telephone': 'Does the household have a telephone (landline or mobile)? (1: Yes, 0: No)',
    'Wealth_Idx_Lb': 'What is the household wealth status? (A higher number indicates a wealthier household)',
    'Tot_child_born': 'What is the total number of children the mother has given birth to?',
    'Sons_died': 'How many sons born to the mother have passed away?',
    'Daughters_died': 'How many daughters born to the mother have passed away?',
    'Curr_Preg': 'Is the mother currently pregnant? (1: Yes, 0: No)',
    'LastChild_Want': 'Was the last child born wanted at the time of conception? (1: Yes, 0: No)',
    'Curr_BrstFeed': 'Is the child currently being breastfed? (1: Yes, 0: No)',
