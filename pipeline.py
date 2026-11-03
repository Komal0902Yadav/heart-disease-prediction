"""
pipeline.py - all the data-mining logic for the Heart Disease project.
(Same steps as the Colab notebook: load -> clean -> encode -> split -> scale -> train -> evaluate)
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
NUM_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CAT_COLS = ['cp', 'restecg', 'slope', 'thal']          # one-hot encoded
INT_COLS = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']


def load_data(path):
    """Load the Cleveland file and apply the preprocessing from the notebook."""
    df = pd.read_csv(path, names=COLS, na_values='?')
    # missing values: mode for the two categorical-like columns that have '?'
    for c in ['ca', 'thal']:
        df[c] = df[c].fillna(df[c].mode()[0])
    df = df.drop_duplicates().reset_index(drop=True)
    df['target'] = (df['target'] > 0).astype(int)       # 0-4 -> binary
    df[INT_COLS] = df[INT_COLS].astype(int)
    return df


def encode(features):
    """One-hot encode the categorical columns. Always returns float columns."""
    return pd.get_dummies(features, columns=CAT_COLS).astype(float)


def train_all(df):
    """Train the 3 models and collect everything the app needs to show."""
    X = encode(df.drop(columns='target'))
    y = df['target']
    columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # scale on TRAIN only, apply to test (no data leakage)
    scaler = MinMaxScaler()
    X_train, X_test = X_train.copy(), X_test.copy()
    X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test[NUM_COLS] = scaler.transform(X_test[NUM_COLS])

    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Naive Bayes': GaussianNB(),
    }

    rows, cms = [], {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, m in models.items():
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        cms[name] = confusion_matrix(y_test, p)
        cv_acc = cross_val_score(make_pipeline(MinMaxScaler(), clone(m)),
                                 X, y, cv=cv, scoring='accuracy').mean()
        rows.append([name,
                     accuracy_score(y_test, p),
                     precision_score(y_test, p),
                     recall_score(y_test, p),
                     f1_score(y_test, p),
                     cv_acc])
    results = pd.DataFrame(rows, columns=['Model', 'Accuracy', 'Precision',
                                          'Recall', 'F1-Score', 'CV Accuracy (5-fold)'])

    importance = pd.Series(models['Random Forest'].feature_importances_,
                           index=columns).sort_values(ascending=False)

    return dict(models=models, scaler=scaler, columns=columns, results=results,
                cms=cms, importance=importance, n_train=len(X_train), n_test=len(X_test))


def predict_proba(bundle, model_name, inputs):
    """Probability of heart disease for ONE patient (dict of the 13 raw features)."""
    row = pd.DataFrame([inputs])
    row[INT_COLS] = row[INT_COLS].astype(int)
    row = encode(row)
    row = row.reindex(columns=bundle['columns'], fill_value=0.0)   # same columns/order as training
    row[NUM_COLS] = bundle['scaler'].transform(row[NUM_COLS])
    return float(bundle['models'][model_name].predict_proba(row)[0][1])
