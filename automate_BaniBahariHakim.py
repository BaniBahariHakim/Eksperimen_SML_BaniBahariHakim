import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import sys


def load_data(filepath):
    return pd.read_csv(filepath, sep=';')


def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)


def encode_target(df):
    df = df.copy()
    df['quality_label'] = df['quality'].apply(lambda q: 0 if q <= 5 else (1 if q <= 7 else 2))
    return df.drop('quality', axis=1)


def scale_features(df, target='quality_label'):
    X = df.drop(target, axis=1)
    y = df[target].reset_index(drop=True)
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    return pd.concat([X_scaled, y], axis=1)


def preprocess(input_path, output_path):
    df = load_data(input_path)
    df = remove_duplicates(df)
    df = encode_target(df)
    df = scale_features(df)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved -> {output_path}  shape: {df.shape}")
    return df


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "winequality-red.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "preprocessing/train.csv"
    preprocess(src, dst)
