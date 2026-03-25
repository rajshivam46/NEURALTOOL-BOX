import pandas as pd
import numpy as np

def generate_datasets():
    print("Generating datasets...")
    
    # 1. Perceptron Dataset (Linearly Separable)
    np.random.seed(42)
    # Class 0: centered around (-2, -2)
    X1 = np.random.randn(100, 2) + np.array([-2, -2])
    # Class 1: centered around (2, 2)
    X2 = np.random.randn(100, 2) + np.array([2, 2])
    
    X_p = np.vstack((X1, X2))
    y_p = np.hstack((np.zeros(100), np.ones(100)))
    
    df_p = pd.DataFrame(X_p, columns=['Feature1', 'Feature2'])
    df_p['Target'] = y_p.astype(int)
    # Shuffle
    df_p = df_p.sample(frac=1).reset_index(drop=True)
    df_p.to_csv('perceptron_dataset.csv', index=False)
    print("Created perceptron_dataset.csv")

    # 2. Backpropagation Dataset (Non-linear, XOR-like pattern)
    X_b = np.random.uniform(-5, 5, (250, 2))
    # Target is 1 if inputs have different signs (XOR quadrants)
    y_b = np.logical_xor(X_b[:, 0] > 0, X_b[:, 1] > 0).astype(int)
    
    df_b = pd.DataFrame(X_b, columns=['Feature1', 'Feature2'])
    df_b['Target'] = y_b
    df_b.to_csv('backprop_dataset.csv', index=False)
    print("Created backprop_dataset.csv")

    # 3. RNN Dataset (Time Series / Sine Wave with Noise)
    t = np.linspace(0, 100, 1000)
    # Sine wave plus slight gaussian noise
    y_r = np.sin(t) + np.random.normal(0, 0.1, 1000)
    
    df_r = pd.DataFrame(y_r, columns=['Value'])
    df_r.to_csv('rnn_dataset.csv', index=False)
    print("Created rnn_dataset.csv")

if __name__ == "__main__":
    generate_datasets()
    print("All datasets generated successfully!")
