import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def safe_svd(mat, rank):
    rank = min(rank, min(mat.shape))
    mat = mat + 1e-10 * np.random.randn(*mat.shape)
    try:
        U, s, Vh = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        from sklearn.utils.extmath import randomized_svd
        U, s, Vh = randomized_svd(mat, n_components=rank, random_state=42)
    r = min(rank, len(s))
    return U[:, :r], s[:r], Vh[:r, :]

def tt_svd(tensor, rank):
    d = tensor.ndim
    cores = []
    current = tensor
    for i in range(d-1):
        n = current.shape[0]
        mat = current.reshape(n, -1)
        U, s, Vh = safe_svd(mat, rank)
        r = len(s)
        if i == 0:
            core = U.reshape(1, n, r)
        else:
            # Reshape using the previous core's right dimension
            core = U.reshape(cores[-1].shape[2], n, r)
        cores.append(core)
        current = np.diag(s) @ Vh
    last_core = current.reshape(cores[-1].shape[2], -1, 1)
    cores.append(last_core)
    return cores

def flatten_cores(cores):
    features = []
    for core in cores:
        features.append(core.flatten())
    return np.concatenate(features)

def build_tensor_for_etf(returns_df, macro_df, etf, window=60):
    ret = returns_df[etf].iloc[-window:].values
    # Rolling functions may return NaN; fill them
    mean5 = returns_df[etf].rolling(5).mean().iloc[-window:].values
    std21 = returns_df[etf].rolling(21).std().iloc[-window:].values
    skew60 = returns_df[etf].rolling(60).skew().iloc[-window:].values
    kurt60 = returns_df[etf].rolling(60).kurt().iloc[-window:].values
    etf_features = np.column_stack([ret, mean5, std21, skew60, kurt60])
    etf_features = np.nan_to_num(etf_features)
    macro = macro_df.iloc[-window:].values
    macro = np.nan_to_num(macro)
    T = np.einsum('ti,tj->tij', etf_features, macro)
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
            target = returns_df[ticker].iloc[i+1]
            # Skip if target is NaN
            if np.isnan(target):
                continue
            X.append(features)
            y.append(target)
        X = np.array(X)
        y = np.array(y)
        if len(X) < 10:
            continue
        # Remove any remaining NaN/inf in X
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
