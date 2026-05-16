import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def build_tensor_for_etf(returns_df, macro_df, etf, window=60):
    """Build 3‑order tensor (time × ETF features × macro features)."""
    # ETF features: daily return, 5d mean, 21d std, 60d skew, 60d kurt
    ret = returns_df[etf].iloc[-window:].values
    etf_features = np.column_stack([
        ret,
        returns_df[etf].rolling(5).mean().iloc[-window:].values,
        returns_df[etf].rolling(21).std().iloc[-window:].values,
        returns_df[etf].rolling(60).skew().iloc[-window:].values,
        returns_df[etf].rolling(60).kurt().iloc[-window:].values
    ])
    # Replace NaN/Inf with 0
    etf_features = np.nan_to_num(etf_features, nan=0.0, posinf=0.0, neginf=0.0)
    # Macro features (levels)
    macro = macro_df.iloc[-window:].values
    macro = np.nan_to_num(macro, nan=0.0, posinf=0.0, neginf=0.0)
    # Outer product to create 3D tensor
    T = np.einsum('ti,tj->tij', etf_features, macro)
    # Flatten all dimensions except the first (time) to get a 2D matrix (time × combined features)
    # Actually we want a single feature vector per sample (time). So we flatten the time dimension? No, we need one feature vector per ETF per day.
    # For training, we will create many such tensors over time. So here we return the flattened tensor (1D) for a single time window.
    # We'll flatten the entire tensor into a vector.
    flat = T.flatten()
    return flat

def train_tensor_predictor(returns_df, macro_df, tickers, window=60, rank=5):
    """
    For each ETF, build a dataset of flattened tensors and use PCA+LinearRegression.
    rank here is the number of PCA components (replaces TT rank).
    """
    models = {}
    for ticker in tickers:
        if ticker not in returns_df.columns:
            continue
        if len(returns_df) < window + 5:
            continue
        X = []
        y = []
        for i in range(window, len(returns_df)-1):
            # Slice returns and macro up to i
            ret_slice = returns_df.iloc[i-window:i]
            macro_slice = macro_df.iloc[i-window:i]
            # Build tensor and flatten
            flat = build_tensor_for_etf(ret_slice, macro_slice, ticker, window)
            X.append(flat)
            y.append(returns_df[ticker].iloc[i+1])
        X = np.array(X)
        y = np.array(y)
        # Remove any rows with NaN in y or X
        nan_mask = ~np.isnan(y)
        X = X[nan_mask]
        y = y[nan_mask]
        if len(X) < 10:
            continue
        # Remove constant columns (variance zero)
        std = X.std(axis=0)
        const_cols = np.where(std == 0)[0]
        if len(const_cols) > 0:
            X = np.delete(X, const_cols, axis=1)
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # PCA for dimensionality reduction
        pca = PCA(n_components=min(rank, X_scaled.shape[1]), random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        # Linear regression
        model = LinearRegression()
        model.fit(X_pca, y)
        models[ticker] = (model, scaler, pca)
    return models

def predict_next_return(models, returns_df, macro_df, ticker, window):
    if ticker not in models:
        return np.nan
    if len(returns_df) < window + 1:
        return np.nan
    ret_slice = returns_df.iloc[-window:]
    macro_slice = macro_df.iloc[-window:]
    flat = build_tensor_for_etf(ret_slice, macro_slice, ticker, window)
    X = flat.reshape(1, -1)
    model, scaler, pca = models[ticker]
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return model.predict(X_pca)[0]
