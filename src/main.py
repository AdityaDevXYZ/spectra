import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os
import joblib

# Create necessary directories
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def perform_eda_and_cleaning(df):
    print("Performing EDA and Data Cleaning...")
    # Target variable distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='koi_disposition')
    plt.title('Distribution of KOI Disposition')
    plt.savefig('reports/figures/target_distribution.png')
    plt.close()

    # Drop non-predictive columns (IDs, names, urls, comments)
    cols_to_drop = [
        'rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_disp_prov', 
        'koi_comment', 'koi_tce_delivname', 'koi_fwm_stat_sig'
    ]
    
    # Drop columns that have too many missing values (e.g., > 50%)
    missing_pct = df.isnull().mean()
    cols_to_drop.extend(missing_pct[missing_pct > 0.5].index.tolist())
    
    # Drop them if they exist in the dataframe
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Separate features and target
    y = df['koi_disposition']
    X = df.drop(columns=['koi_disposition', 'koi_pdisposition'], errors='ignore') # koi_pdisposition might be leaky

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(exclude=['number']).columns

    # Fill missing numeric values with median
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    # Fill missing categorical values with mode
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # One-hot encode categorical features if any exist
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    print(f"Final feature shape after cleaning: {X.shape}")
    return X, y

def train_and_evaluate(X, y):
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)

    print("Evaluating Model...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('reports/figures/confusion_matrix.png')
    plt.close()

    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:] # Top 20 features
    plt.figure(figsize=(10, 8))
    plt.title("Top 20 Feature Importances")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.savefig('reports/figures/feature_importance.png')
    plt.close()

    # Save metrics to a file
    with open('reports/metrics.txt', 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, 'models/random_forest_model.pkl')
    print("Model and metrics saved successfully.")

if __name__ == "__main__":
    filepath = 'data/KOI_Cumulative_clean.csv'
    df = load_data(filepath)
    X, y = perform_eda_and_cleaning(df)
    train_and_evaluate(X, y)
