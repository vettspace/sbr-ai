"""Feature engineering."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def safe_divide(a, b):
    return np.divide(
        a, b, out=np.full_like(a, np.nan, dtype=float), where=(b != 0)
    )


def feature_engineering(df, is_training=True):
    target = None
    if is_training and 'Churn' in df.columns:
        target = df['Churn']
        df = df.drop('Churn', axis=1)

    df = df.replace({True: 1, False: 0})

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'NumTransactions' in df.columns and 'Tenure' in df.columns:
        df['TotalTransactions'] = df['NumTransactions'] * df['Tenure']
        df['AvgTransactionsPerMonth'] = safe_divide(
            df['NumTransactions'], df['Tenure']
        )

    if 'TotalCharges' in df.columns and 'NumTransactions' in df.columns:
        df['ChargeToTransactionRatio'] = safe_divide(
            df['TotalCharges'], df['NumTransactions']
        )

    if 'TotalCharges' in df.columns:
        df['HighValueCustomer'] = (
            df['TotalCharges'] > df['TotalCharges'].median()
        ).astype(int)

    if 'Tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['TenureByCharges'] = df['Tenure'] * df['MonthlyCharges']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.fillna(0, inplace=True)

    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)

    if is_training and target is not None:
        df['Churn'] = target

    return df


if __name__ == "__main__":
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')
    X_train['Churn'] = y_train['Churn']

    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')
    X_test['Churn'] = y_test['Churn']

    X_train = feature_engineering(X_train, is_training=True)
    X_test = feature_engineering(X_test, is_training=True)

    y_train = X_train['Churn']
    X_train = X_train.drop('Churn', axis=1)

    y_test = X_test['Churn']
    X_test = X_test.drop('Churn', axis=1)

    X_train.to_csv('data/X_train_fe.csv', index=False)
    X_test.to_csv('data/X_test_fe.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
