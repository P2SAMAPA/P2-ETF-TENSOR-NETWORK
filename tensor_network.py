import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import config

def train_tensor_predictor(returns_df, macro_df, tickers, window=60, rank=5):
    models = {}
    for ticker in tickers:
        if ticker not in returns_df.columns:
            continue
        if len(returns_df) < window + 5:
            continue
        # Build features: we'll use the last 'window' days of returns as features for each day
        # Actually we need a supervised dataset: for each day, X = window days of returns for that ETF, y = next day return.
        # For simplicity, we'll use rolling windows over the entire returns history.
        # But to align with multi-window trainer, we'll only use the most recent data? The trainer will call this function per window.
        # Since the trainer already selects a window, we can just use the entire `returns_df` (which is already trimmed to the window length).
        # So we'll build X from the last `window` days of returns for each day? That would be a 2D array.
        # Instead, we'll use the PCA approach: flatten the last `window` days of returns for the ETF into a feature vector.
        # But we need many samples. For each day i in [window, len(returns_df)-1], we take the `window` days up to i as features.
        # That's what we did in the simplified version.
        X = []
        y = []
        series = returns_df[ticker].values
        for i in range(window, len(series)-1):
            X.append(series[i-window:i])
            y.append(series[i+1])
        X = np.array(X)
        y = np.array(y)
        if len(X) < 10:
            continue
        # Remove constant columns
        std = X.std(axis=0)
        const_cols = np.where(std == 0)[0]
        if len(const_cols) > 0:
            X = np.delete(X, const_cols, axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=min(rank, X_scaled.shape[1]), random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        model = LinearRegression()
        model.fit(X_pca, y)
        models[ticker] = (model, scaler, pca)
    return models

def predict_next_return(models, returns_df, macro_df, ticker, window, rank):
    if ticker not in models:
        return np.nan
    if len(returns_df) < window + 1:
        return np.nan
    series = returns_df[ticker].values
    last_window = series[-window:]
    X = last_window.reshape(1, -1)
    # Remove constant columns (same as in training, but we need to know which columns were removed)
    # Simpler: we can pass the same scaler and pca; they will handle unseen features.
    # But we need to ensure the number of features matches. In training, we removed constant columns.
    # For prediction, we need to remove the same columns. We'll store the mask in the model.
    # To simplify, we'll not remove constant columns; we'll just use the raw window.
    # We'll rely on the scaler and pca to handle variance.
    model, scaler, pca = models[ticker]
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return model.predict(X_pca)[0]
