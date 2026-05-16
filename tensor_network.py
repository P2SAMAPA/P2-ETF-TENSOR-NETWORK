import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def tt_svd(tensor, rank):
    """
    Compute Tensor Train decomposition using sequential SVD.
    Input: numpy ndarray (order d)
    Output: list of TT-cores (each 3‑dimensional)
    """
    d = tensor.ndim
    cores = []
    current = tensor
    for i in range(d-1):
        # Reshape to matrix
        n = current.shape[0]
        mat = current.reshape(n, -1)
        # SVD with rank truncation
        U, s, Vh = np.linalg.svd(mat, full_matrices=False)
        r = min(rank, len(s))
        U = U[:, :r]
        s = s[:r]
        Vh = Vh[:r, :]
        # Reshape U to core (r_left, n, r_right) with r_left = U.shape[0]? Actually U shape is (n, r)
        # For first core, left dimension = 1
        if i == 0:
            core = U.reshape(1, n, r)
        else:
            # Reshape to include previous core's right dimension
            core = U.reshape(cores[-1].shape[2], n, r)
        cores.append(core)
        # Update current for next iteration
        current = np.diag(s) @ Vh
    # Last core: reshape to (r, last_dim, 1)
    last_core = current.reshape(cores[-1].shape[2], -1, 1)
    cores.append(last_core)
    return cores

def flatten_cores(cores):
    """Flatten all TT-cores into a single feature vector."""
    features = []
    for core in cores:
        features.append(core.flatten())
    return np.concatenate(features)

def build_tensor_for_etf(returns_df, macro_df, etf, window=60):
    """
    Build a 3‑order tensor for one ETF:
    - mode 0: time steps (last `window` days)
    - mode 1: ETF features (return, volatility, skew, kurt, etc.)
    - mode 2: macro features (levels or changes)
    Returns a 3D numpy array of shape (window, n_etf_features, n_macro_features)
    """
    # ETF features: daily return, 5‑day mean, 21‑day std, skew (60d), kurt (60d)
    ret = returns_df[etf].iloc[-window:].values
    # For simplicity, use only return as ETF feature? Let's add some.
    # Actually we need a 2D matrix per time. We'll create a 2D array: (window, n_features)
    # Then we'll use macro as third dimension? That's heavy. We'll do: time × ETF_features × macro_features.
    # But macro_features are not time‑varying per day? They are.
    # Let's simplify: build a 2D matrix (window × features) where features = ETF features + macro features.
    # Then apply TT‑SVD on a 2D matrix (order 2). That's not interesting.
    # To have order ≥3, we need to separate ETF and macro as two dimensions. We'll create a 3D tensor:
    # For each day, we have a product of ETF feature vector and macro feature vector (outer product).
    # So the tensor is: T[t, i, j] = ETF_feature[t, i] * macro_feature[t, j]
    # That gives a 3‑order tensor.
    # Compute ETF features over the window: matrix (window, n_etf_feat)
    etf_features = np.column_stack([
        ret,                                    # daily return
        returns_df[etf].rolling(5).mean().iloc[-window:].values,
        returns_df[etf].rolling(21).std().iloc[-window:].values,
        returns_df[etf].rolling(60).skew().iloc[-window:].values,
        returns_df[etf].rolling(60).kurt().iloc[-window:].values
    ])
    # Macro features: use levels and changes
    macro = macro_df.iloc[-window:].values
    # Add a constant? Not needed.
    # Build outer product: T[t,i,j] = etf_features[t,i] * macro[t,j]
    T = np.einsum('ti,tj->tij', etf_features, macro)
    return T

def train_tensor_predictor(returns_df, macro_df, tickers, window=60, rank=10):
    """
    For each ETF, build the 3D tensor, compress via TT‑SVD, then train a linear regression
    from the flattened TT‑cores to next‑day return. Returns a dict with model and scaler per ETF.
    """
    models = {}
    for ticker in tickers:
        if ticker not in returns_df.columns:
            continue
        if len(returns_df) < window + 5:
            continue
        # Build tensor for many days? We need a sequence of tensors for training.
        # Actually we need to create a dataset: for each day i (window..end), build tensor up to i, compress, predict next day.
        # That would be very slow. To keep it practical, we build a single tensor for the last `window` days and use it to predict next day.
        # But for training we need many samples. Let's build a sliding window over days.
        X = []
        y = []
        for i in range(window, len(returns_df)-1):
            # slice returns and macro up to i
            ret_slice = returns_df.iloc[i-window:i]
            macro_slice = macro_df.iloc[i-window:i]
            # build tensor
            T = build_tensor_for_etf(ret_slice, macro_slice, ticker, window)
            # compress via TT‑SVD
            cores = tt_svd(T, rank)
            features = flatten_cores(cores)
            X.append(features)
            # target: next day return of the ETF
            y.append(returns_df[ticker].iloc[i+1])
        X = np.array(X)
        y = np.array(y)
        if len(X) < 10:
            continue
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        models[ticker] = (model, scaler)
    return models

def predict_next_return(models, returns_df, macro_df, ticker, window, rank):
    """Predict next day return using the trained model for the ETF."""
    if ticker not in models:
        return np.nan
    if len(returns_df) < window + 1:
        return np.nan
    # Build tensor for the last `window` days
    ret_slice = returns_df.iloc[-window:]
    macro_slice = macro_df.iloc[-window:]
    T = build_tensor_for_etf(ret_slice, macro_slice, ticker, window)
    cores = tt_svd(T, rank)
    features = flatten_cores(cores).reshape(1, -1)
    model, scaler = models[ticker]
    X_scaled = scaler.transform(features)
    return model.predict(X_scaled)[0]
