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
