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

