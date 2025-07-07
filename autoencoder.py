from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def train_autoencoder(X):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X.shape[1],)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(X.shape[1], activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, X, epochs=10, batch_size=64, verbose=0)

    reconstructions = model.predict(X)
    mse = np.mean(np.power(X - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 95)

    return (mse > threshold).astype(int)
