import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import config

def build_tensor_for_etf(returns_df, macro_df, etf, window=60):
    """Build flattened tensor of fixed size."""
    # ETF features (5)
    ret = returns_df[etf].iloc[-window:].values
    etf_features = np.column_stack([
        ret,
        returns_df[etf].rolling(5).mean().iloc[-window:].values,
        returns_df[etf].rolling(21).std().iloc[-window:].values,
        returns_df[etf].rolling(60).skew().iloc[-window:].values,
        returns_df[etf].rolling(60).kurt().iloc[-window:].values
    ])
    etf_features = np.nan_to_num(etf_features, nan=0.0)
    # Macro features: we'll use the same macro columns consistently
    # If macro_df has missing columns, we'll fill with zeros
    # We'll assume macro_df has columns from config.MACRO_COLUMNS (if defined) or a fixed set
    # For now, use all macro columns present in the data frame; but we need consistency.
    # Let's ensure that the macro_df we use in training and prediction has the same columns.
    macro = macro_df.iloc[-window:].values
    if macro.shape[1] == 0:
        macro = np.zeros((window, 1))  # dummy
    # Outer product: (window, 5) × (window, M) -> (window, 5, M)
    T = np.einsum('ti,tj->tij', etf_features, macro)
    # Flatten to 1D
    flat = T.flatten()
    return flat

def train_tensor_predictor(returns_df, macro_df, tickers, window=60, rank=10):
    """
    For each ETF, build dataset of flattened tensors, apply PCA, then linear regression.
    """
    models = {}
    # Ensure we use a consistent macro column set (the current macro_df's columns)
    # But macro_df may change over time? We'll store the macro columns used during training.
    # For simplicity, we'll use the first valid macro_df we encounter; but we need all macro data to have same shape.
    # We'll just rely on the macro_df passed; it should be consistent across calls.
    # However, in rolling windows, macro_df columns may be the same because they come from the same source.
    for ticker in tickers:
        if ticker not in returns_df.columns:
            continue
        if len(returns_df) < window + 5:
            continue
        X = []
        y = []
        for i in range(window, len(returns_df)-1):
            ret_slice = returns_df.iloc[i-window:i]
            macro_slice = macro_df.iloc[i-window:i]
            # If macro_slice has no columns, add a dummy
            if macro_slice.shape[1] == 0:
                macro_slice = pd.DataFrame(0, index=ret_slice.index, columns=['dummy'])
            flat = build_tensor_for_etf(ret_slice, macro_slice, ticker, window)
            X.append(flat)
            y.append(returns_df[ticker].iloc[i+1])
        X = np.array(X)
        y = np.array(y)
        # Remove NaN in y
        nan_mask = ~np.isnan(y)
        X = X[nan_mask]
        y = y[nan_mask]
        if len(X) < 10:
            continue
        # Remove constant columns
        std = X.std(axis=0)
        non_const = np.where(std > 1e-8)[0]
        X = X[:, non_const]
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # PCA
        n_components = min(rank, X_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        # Regression
        model = LinearRegression()
        model.fit(X_pca, y)
        # Store along with the column indices used (to ensure same features in prediction)
        models[ticker] = (model, scaler, pca, non_const)
    return models

def predict_next_return(models, returns_df, macro_df, ticker, window):
    if ticker not in models:
        return np.nan
    if len(returns_df) < window + 1:
        return np.nan
    ret_slice = returns_df.iloc[-window:]
    macro_slice = macro_df.iloc[-window:]
    if macro_slice.shape[1] == 0:
        macro_slice = pd.DataFrame(0, index=ret_slice.index, columns=['dummy'])
    flat = build_tensor_for_etf(ret_slice, macro_slice, ticker, window)
    X = flat.reshape(1, -1)
    model, scaler, pca, non_const = models[ticker]
    # Select the same constant columns as during training
    X = X[:, non_const]
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return model.predict(X_pca)[0]
