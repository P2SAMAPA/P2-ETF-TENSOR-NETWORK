import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def train_tensor_predictor(returns_df, macro_df, tickers, window=60, rank=5):
    """
    Train a PCA+linear regression model for each ETF using the last `window` days of its returns as features.
    """
    models = {}
    for ticker in tickers:
        if ticker not in returns_df.columns:
            continue
        if len(returns_df) < window + 1:
            continue
        # Build sliding window dataset
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
        # Remove columns with zero variance (constant features)
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
        # Store the mask of removed constant columns to apply later
        models[ticker] = (model, scaler, pca, const_cols)
    return models

def predict_next_return(models, returns_df, macro_df, ticker, window, rank):
    if ticker not in models:
        return np.nan
    series = returns_df[ticker].values
    if len(series) < window:
        return np.nan
    model, scaler, pca, const_cols = models[ticker]
    last_window = series[-window:].reshape(1, -1)
    # Remove constant columns (same as during training)
    if len(const_cols) > 0:
        last_window = np.delete(last_window, const_cols, axis=1)
    X_scaled = scaler.transform(last_window)
    X_pca = pca.transform(X_scaled)
    return model.predict(X_pca)[0]
