# Installer les paquets nécessaires

import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ydata_profiling import ProfileReport
import joblib

# Importez vos données et effectuez la phase d'exploration des données de base

df = pd.read_csv('Financial_inclusion_dataset.csv')
print(df.head())

# Afficher des informations générales sur l'ensemble de données

print(df.shape)
print(df.info())
print(df.describe())
print(df.describe(include='object'))

# Créer un rapport de profilage pandas pour obtenir des informations sur le jeu de données

profile = ProfileReport(df, title="Financial Inclusion Dataset Profiling Report")
profile.to_file("financial_inclusion_report.html")

# Traiter les valeurs manquantes et corrompues

print(df.isnull().sum())

# Supprimer les doublons, s'ils existent

num_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows found: {num_duplicates}")

# Traiter les valeurs aberrantes, si elles existent
# Identify the numerical columns in the DataFrame to prepare for outlier detection using box plots.

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
print("Numerical columns:", numerical_cols.tolist())
# Generate box plots for each numerical column to visualize the distribution and identify potential outliers.

for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Box plot of {col}')
    plt.show()

# Handle outliers in 'household_size' by capping
q1_size = df['household_size'].quantile(0.25)
q3_size = df['household_size'].quantile(0.75)
iqr_size = q3_size - q1_size
upper_bound_size = q3_size + 1.5 * iqr_size
df['household_size'] = df['household_size'].clip(upper=upper_bound_size)

# Handle outliers in 'age_of_respondent' by capping
q1_age = df['age_of_respondent'].quantile(0.25)
q3_age = df['age_of_respondent'].quantile(0.75)
iqr_age = q3_age - q1_age
upper_bound_age = q3_age + 1.5 * iqr_age
df['age_of_respondent'] = df['age_of_respondent'].clip(upper=upper_bound_age)

# Verify the capping by plotting box plots again
for col in ['household_size', 'age_of_respondent']:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Box plot of {col} after outlier handling')
    plt.show()

# Encoder les caractéristiques catégorielles

categorical_cols = df.select_dtypes(include='object').columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print(df_encoded.head())
print(df_encoded.shape)

# Former et tester un classificateur d'apprentissage automatique

# Separate features (X) and target variable (y)
X = df_encoded.drop('bank_account_Yes', axis=1)
y = df_encoded['bank_account_Yes']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

import pickle
    
# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the list of feature names
with open('features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("Model and features saved successfully.")
