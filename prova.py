import numpy as np

def create_sequences_mv(data, window_size=5, horizon=1, step=1):
    """
    Crea dataset sliding window da serie multivariata.
    Parametri:
        data: ndarray di shape (n_samples, n_features)
        window_size: lunghezza input
        horizon: passi futuri da predire
        step: scorrimento finestra
    Ritorna:
        X: (n_seq, window_size, n_features)
        y: (n_seq, horizon, n_features)
    """
    data = np.asarray(data)
    n_samples = (len(data) - window_size - horizon + 1) // step
    n_features = data.shape[1]

    X = np.empty((n_samples, window_size, n_features))
    y = np.empty((n_samples, horizon))

    for i in range(n_samples):
        start = i * step
        X[i] = data[start : start + window_size]
        y[i] = data[start + window_size : start + window_size + horizon]

    if horizon == 1:
        y = y[:, 0, :]  # shape: (n_seq, n_features)

    return X, y


data = np.random.rand(1000, 8)  # 8 indicatori tecnici
X, y = create_sequences_mv(data, window_size=5, horizon=1, step=1)

print(X.round(2))  # (966, 30, 8)
print(y.shape)