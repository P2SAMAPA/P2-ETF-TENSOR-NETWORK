import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def safe_svd(mat, rank):
    """
    Perform SVD with regularisation and rank truncation.
    """
    # Add small regularisation to the matrix to avoid singularities
    mat_reg = mat + 1e-8 * np.eye(mat.shape[0])[:mat.shape[0], :mat.shape[1]]  # not ideal; better: add small noise
    # Alternative: add a small identity to the covariance-like matrix? We'll just add tiny Gaussian noise
    mat = mat + 1e-8 * np.random.randn(*mat.shape)
    try:
        U, s, Vh = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        # If still fails, use randomised SVD via sklearn
        from sklearn.utils.extmath import randomized_svd
        U, s, Vh = randomized_svd(mat, n_components=min(rank, min(mat.shape)), random_state=42)
        return U[:, :rank], s[:rank], Vh[:rank, :]
    # Truncate
    r = min(rank, len(s))
    return U[:, :r], s[:r], Vh[:r, :]

def tt_svd(tensor, rank):
    """
    Tensor Train decomposition using sequential SVD with rank truncation.
    """
    d = tensor.ndim
    cores = []
    current = tensor
    for i in range(d-1):
        # Reshape to matrix
        n = current.shape[0]
        mat = current.reshape(n, -1)
        U, s, Vh = safe_svd(mat, rank)
        r = len(s)
        # Build core
        if i == 0:
            core = U.reshape(1, n, r)
        else:
            core = U.reshape(cores[-1].shape[2], n, r)
        cores.append(core)
        # Update current
        current = np.diag(s) @ Vh
    # Last core
    last_core = current.reshape(cores[-1].shape[2], -1, 1)
    cores.append(last_core)
    return cores

def flatten_cores(cores):
    features = []
    for core in cores:
        features.append(core.flatten())
    return np.concatenate(features)

def build_tensor_for_etf(returns_df, macro_df, etf, window=60):
    """
    Build a 3‑order tensor (time × ETF features × macro features).
    Returns a 3D numpy array.
    """
    # ETF features (per day)
    ret = returns_df[etf].iloc[-window:].values
    etf_features = np.column_stack([
        ret,
        returns_df[etf].rolling(5).mean().iloc[-window:].values,
        returns_df[etf].rolling(21).std().iloc[-window:].values,
        returns_df[etf].rolling(60).skew().iloc[-window:].values,
        returns_df[etf].rolling(60).kurt().iloc[-window:].values
    ])
    # Replace any NaN with 0
    etf_features = np.nan_to_num(etf_features)
    # Macro features (levels)
    macro = macro_df.iloc[-window:].values
    macro = np.nan_to_num(macro)
    # Outer product over time
    T = np.einsum('ti,tj->tij', etf_features, macro)
    # Add a small constant to avoid all‑zero tensors
    T = T + 1e-8
    return T

def train_tensor_predictor(returns_df, macro_df, tickers, window=60, rank=5):
    models = {}
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
            T = build_tensor_for_etf(ret_slice, macro_slice, ticker, window)
            cores = tt_svd(T, rank)
            features = flatten_cores(cores)
            X.append(features)
            y.append(returns_df[ticker].iloc[i+1])
        X = np.array(X)
        y = np.array(y)
        if len(X) < 10:
            continue
        # Remove any remaining NaN/inf
        X = np.nan_to_num(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        models[ticker] = (model, scaler)
    return models

def predict_next_return(models, returns_df, macro_df, ticker, window, rank):
    if ticker not in models:
        return np.nan
    if len(returns_df) < window + 1:
        return np.nan
    ret_slice = returns_df.iloc[-window:]
    macro_slice = macro_df.iloc[-window:]
    T = build_tensor_for_etf(ret_slice, macro_slice, ticker, window)
    cores = tt_svd(T, rank)
    features = flatten_cores(cores).reshape(1, -1)
    model, scaler = models[ticker]
    X_scaled = scaler.transform(features)
    return model.predict(X_scaled)[0]
