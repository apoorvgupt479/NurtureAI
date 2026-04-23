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
    'ChildFood_bottle': 'Was the child fed using a bottle with a nipple? (1: Yes, 0: No)',
    'Child_putToBrst': 'Was the child put to the breast immediately after birth (within one hour)? (1: Yes, 0: No)',
    'Resp_weight': "What is the mother's weight in kilograms (kg)?",
    'Resp_height': "What is the mother's height in meters (m)?",
    'Hg_levelAdjusted': "What is the mother's hemoglobin level (adjusted for altitude, in g/dl)?",
    'Anemia_level': 'What is the severity of anemia? (0: No anemia, 1: Mild, 2: Moderate, 3: Severe)',
    'HealthInsurance': 'Is the family covered by any health insurance scheme? (1: Yes, 0: No)',
    'B_ChildTwin': 'Was the child part of a multiple birth (e.g., twins, triplets)? (1: Yes, 0: No)',
    'Birth_Order': 'What is the birth order of the child (e.g., 1st, 2nd, etc.)?',
    'Birth_Size': 'What was the perceived size of the child at birth? (A higher number indicates a larger or healthier size)',
    'Birth_Weight': 'What was the weight of the child at birth in grams (g)?',
    'Delivery_CSection': 'Was the child delivered via C-section (Cesarean section)? (1: Yes, 0: No)',
    'Preg_iron': 'Did the mother take iron tablets or syrup during pregnancy? (1: Yes, 0: No)',
    'Preg_intParaDrug': 'Did the mother take intestinal parasite drugs during pregnancy? (1: Yes, 0: No)',
    'Preg_Complication': 'Did the mother experience any complications during pregnancy? (1: Yes, 0: No)',
    'Antenatal_visit': 'What is the total number of antenatal care (ANC) visits during pregnancy?',
    'HepatitisB_atBirth': 'Did the child receive the Hepatitis B vaccine at birth? (1: Yes, 0: No) - Crucial for protecting against liver disease.',
    'VitaminA': 'Did the child receive a Vitamin A dose in the last 6 months? (1: Yes, 0: No) - Important for immunity and vision.',
    'IronPill': 'Did the child receive iron pills or syrup? (1: Yes, 0: No) - Helps prevent anemia.',
    'IntestinalDrug': 'Did the child receive drugs for intestinal worms? (1: Yes, 0: No) - Important for nutrient absorption and growth.',
    'Diarrhea': 'Has the child had diarrhea recently (in the last 2 weeks)? (1: Yes, 0: No)',
    'Fever': 'Has the child had a fever recently (in the last 2 weeks)? (1: Yes, 0: No)',
    'ShortBreaths': 'Has the child experienced rapid or short breaths (possible symptom of pneumonia) recently? (1: Yes, 0: No)',
    'Birth_Month': "What is the child's month of birth?",
    'Birth_Year': "What is the child's year of birth?",
    'ChildAge_mnths': "What is the child's age in months?",
    'Married_age': "What was the mother's age at her first marriage?",
    'Alcohol': 'Does the mother consume alcohol? (1: Yes, 0: No)',
    'Smoke_atHome': 'Does anyone smoke inside the house? (1: Yes, 0: No)',
    'Preg_months': 'What was the duration of the pregnancy in completed months?',
    'First3Day_other': 'Was the child given any other liquids in the first 3 days after birth, besides breast milk or water? (1: Yes, 0: No)',
    'First3Day_janamGhutti': 'Was the child given Janam Ghutti (traditional herbal preparation) in the first 3 days after birth? (1: Yes, 0: No)',
    'First3Day_honey': 'Was the child given honey in the first 3 days after birth? (1: Yes, 0: No)',
    'First3Day_gripeWater': 'Was the child given gripe water in the first 3 days after birth? (1: Yes, 0: No)',
    'First3Day_fruitJuice': 'Was the child given fruit juice in the first 3 days after birth? (1: Yes, 0: No)',
    'First3Day_infFormu': 'Was the child given infant formula in the first 3 days after birth? (1: Yes, 0: No)',
    'First3Day_animalMilk': 'Was the child given animal milk (cow, buffalo, etc.) in the first 3 days after birth? (1: Yes, 0: No)',
    'First3Day_sugarWater': 'Was the child given sugar or glucose water in the first 3 days after birth? (1: Yes, 0: No)',
    'First3Day_plainWater': 'Was the child given plain water in the first 3 days after birth? (1: Yes, 0: No)',
    'First3Day_saltSol': 'Was the child given salt solution in the first 3 days after birth? (1: Yes, 0: No)',
    'Hypertension': 'Does the mother have high blood pressure (hypertension)? (1: Yes, 0: No)',
    'Diabetes': 'Does the mother have diabetes? (1: Yes, 0: No)',
    'Thyroid': 'Does the mother have thyroid problems? (1: Yes, 0: No)',
    'RespDisease': 'Does the mother have respiratory diseases (e.g., Asthma)? (1: Yes, 0: No)',
    'HeartDisease': 'Does the mother have heart disease? (1: Yes, 0: No)',
    'Cancer': 'Has the mother been diagnosed with cancer? (1: Yes, 0: No)',
    'Kidney': 'Does the mother have kidney disease? (1: Yes, 0: No)',
    'ultrasound': 'Was an ultrasound performed during pregnancy? (1: Yes, 0: No)',
    'PostnatalChk': 'Did the mother receive a postnatal health check (after delivery)? (1: Yes, 0: No)',
    'Resp_healthChk': 'Did the mother receive a general health check-up in the last year? (1: Yes, 0: No)',
    'DPTB': 'Did the child receive the DPT booster shot? (1: Yes, 0: No) - Boosts protection against Diphtheria, Pertussis, and Tetanus.',
    'MMR': 'Did the child receive the MMR (Measles, Mumps, Rubella) vaccine? (1: Yes, 0: No) - Protects against these common childhood diseases.',
    'Benefit_HCare': 'Has the family benefited from any government health schemes? (1: Yes, 0: No)',
    'Smoke': 'Does the mother smoke? (1: Yes, 0: No)',
    'Betel_Leaf': 'Does the mother chew betel leaf or tobacco? (1: Yes, 0: No)',
    'Tobacco': 'Does the mother use any form of tobacco? (1: Yes, 0: No)',
    'Prenatal_care': 'Did the mother receive any prenatal care (before birth)? (1: Yes, 0: No)',
    'Breastfeed_duration': 'What is the duration of breastfeeding in months?',
    'B_ChildSex_Male': 'Is the child male? (1: Yes, 0: No)',
    'Curr_MaritalStatus_Single Parent': 'Is the mother a single parent? (1: Yes, 0: No)',
    'DPT_full': 'Did the child receive the full course of DPT vaccines? (1: Yes, 0: No) - Essential for full protection against Diphtheria, Pertussis, and Tetanus.',
    'MEASLES_full': 'Did the child receive the full course of Measles vaccines? (1: Yes, 0: No) - Crucial for preventing Measles, a serious childhood illness.',
    'JE_full': 'Did the child receive the full course of Japanese Encephalitis vaccines? (1: Yes, 0: No) - Protects against a dangerous brain infection.',
    'Religion': "What is the mother's religion?",
    'Ethnicity': "What is the mother's ethnicity?",
    'Water_Source': 'What is the primary source of drinking water for the household?',
    'DeliveryPlace': 'Where was the delivery performed?'
}

def get_label(col):
    """Retrieves the human-friendly label for a given feature column."""
    return feature_labels.get(col, col.replace('_', ' '))

# --- CELL ---

def create_input_for_prediction(user_data, original_feature_columns, model_expected_columns):
    """Prepares user input for prediction, handling one-hot encoding for 'State' and other grouped categories."""
    # 1. Create a template DataFrame with all model_expected_columns, initialized to 0
    input_df_template = pd.DataFrame(0, index=[0], columns=model_expected_columns)

    # Define wealth mapping here as it's specific to input processing
    wealth_mapping = {
        'Poor': 0,
        'Middle Class': 2,
        'Rich': 4
    }

    # 2. Iterate through user_data and populate the template
    for key, value in user_data.items():
        if key == 'Religion':
            # Handle one-hot encoding for Religion
            # If 'Hindu' is selected, no 'Religion_X' column needs to be 1 (they all stay 0 by default)
            if value != 'Hindu':
                col_name = f'Religion_{value}'
                if col_name in model_expected_columns:
                    input_df_template.loc[0, col_name] = 1
        elif key == 'Ethnicity':
            # Handle one-hot encoding for Ethnicity
            col_name = f'Ethnicity_{value}'
            if col_name in model_expected_columns:
                input_df_template.loc[0, col_name] = 1
        elif key == 'Water_Source':
            # Handle one-hot encoding for Water_Source
            # Note: the .replace(' ', '') was in original code, maintaining that for consistency
            col_name = f'Water_Source_{value.replace(' ', '')}'
            if col_name in model_expected_columns:
                input_df_template.loc[0, col_name] = 1
        elif key == 'DeliveryPlace':
            # Handle one-hot encoding for DeliveryPlace
            col_name = f'DeliveryPlace_{value}'
            if col_name in model_expected_columns:
                input_df_template.loc[0, col_name] = 1
        elif key == 'State':
            # Handle one-hot encoding for State
            col_name = f'State_{value}'
            if col_name in model_expected_columns:
                input_df_template.loc[0, col_name] = 1
        elif key == 'Wealth_Idx_Lb':
            # Map the string value from user_data to numerical and set
            if key in model_expected_columns:
                input_df_template.loc[0, key] = wealth_mapping.get(value, 2) # Default to Middle Class (2)
        else:
            # For direct features, simply assign the value if the column is expected by the model
            if key in model_expected_columns:
                input_df_template.loc[0, key] = value

    # The input_df_template already has the correct columns and order,
    # so it can be directly returned.
    return input_df_template

def generate_recommendations(user_data, top_features_df):
    """Generates rule-based recommendations based on top features and user input.
    Args:
        user_data (dict): Dictionary of user inputs.
        top_features_df (pd.DataFrame): DataFrame containing top features and their coefficients.
    Returns:
        str: A string of rule-based recommendations.
    """
    recommendations = []
    for index, row in top_features_df.iterrows():
        feature = row['Feature']

        base_feature = feature
        if feature.startswith('Religion_'):
            base_feature = 'Religion'
        elif feature.startswith('Ethnicity_'):
            base_feature = 'Ethnicity'
        elif feature.startswith('Water_Source_') and feature != 'Water_Source_Time':
            base_feature = 'Water_Source'
        elif feature.startswith('DeliveryPlace_'):
            base_feature = 'DeliveryPlace'
        elif feature.startswith('State_'):
             base_feature = 'State'

        if base_feature in user_data:
            user_value = user_data[base_feature]
            label = feature_labels.get(base_feature, base_feature.replace('_', ' '))

            if base_feature in ['Religion', 'Ethnicity', 'Water_Source', 'DeliveryPlace']:
                pass
            elif user_value == 0:
                if 'DPT' in label or 'MEASLES' in label or 'JE' in label or 'HepatitisB' in label or 'MMR' in label:
                    recommendations.append(f"Ensure child receives full vaccination course for {label}. Vaccinations are crucial for protecting children from severe diseases.")
                elif 'Breastfeed' in label or 'BrstFeed' in label:
                    recommendations.append(f"Encourage exclusive breastfeeding for the first 6 months. It provides vital nutrients and antibodies for the baby's health.")
                elif 'Toilet' in label or 'Water_Source' in label:
                    recommendations.append(f"Improve sanitation and access to safe drinking water, as indicated by '{label}' not being optimal. Clean environments prevent many childhood illnesses.")
                elif 'Preg' in label or 'Antenatal' in label or 'Postnatal' in label or 'Resp_healthChk' in label:
                    recommendations.append(f"Prioritize maternal health check-ups (e.g., '{label}') during and after pregnancy. A healthy mother is key to a healthy child.")
                elif 'House_' in label:
                    recommendations.append(f"Consider improving household amenities like '{label}'. These can indirectly impact health and living conditions.")
                elif 'Wealth_Idx_Lb' in base_feature:
                    recommendations.append(f"Explore programs that can help improve economic stability and access to resources, as current wealth index is low.")
                else:
                    recommendations.append(f"Consider addressing factors related to '{label}' as they might contribute to health risks.")
        elif feature.startswith('State_'):
            pass

    if not recommendations:
        return "No specific rule-based recommendations found based on your input and top features. Keep up the good work!"

    return "\n".join([f"- {rec}" for rec in recommendations])

# --- CELL ---

llm_available = False
try:
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    llm_model = genai.GenerativeModel('gemini-2.5-flash')
    llm_available = True
    print("Gemini API configured successfully.")
except Exception as e:
    print("Note: Gemini API key not found or configuration failed.")
    print("Please add GOOGLE_API_KEY to Colab Secrets for LLM recommendations or check your API key.")
    llm_available = False

def get_llm_recommendations(user_data, rule_recs, prediction_status):
    """Generates recommendations using google.generativeai.GenerativeModel."""
    if not llm_available:
        return "LLM recommendations are unavailable. Please check your API key."

    prompt = f"""
    You are a supportive maternal and child health advisor.
    Based on the following data for a mother and child in India:
    - Mother's Age: {user_data.get('Res_Age', 'N/A')}
    - State: {user_data.get('State', 'N/A')}
    - Prediction Status: {prediction_status}
    - Technical Rule-Based Advice: {rule_recs}

    The user has also answered the following details (1=Yes, 0=No):
    { {k:v for k,v in user_data.items() if k not in ['Res_Age', 'State']} }

    Please provide a warm, easy-to-understand, and personalized set of health recommendations.
    Focus on encouragement and explain clearly why certain actions (like breastfeeding or vaccination) are vital based on their specific 'No' answers. Avoid technical jargon.
    Keep it concise (around 8-10 lines).
    """
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not generate LLM response: {str(e)}"

# --- CELL ---

def on_button_click_enhanced(b):
    with output_area:
        output_area.clear_output()
        current_widget_values = {name: widget.value for name, widget in input_widgets.items()}

        input_df = create_input_for_prediction(current_widget_values, original_feature_columns, model_expected_columns)

        prediction = selected_model.predict(input_df)[0]
        probs = selected_model.predict_proba(input_df)[0]
        status_label = "Healthy/Alive" if prediction == 1 else "At Risk"

        status_html = f"<b style='color:green;'>{status_label}</b>" if prediction == 1 else f"<b style='color:red;'>{status_label}</b>"
