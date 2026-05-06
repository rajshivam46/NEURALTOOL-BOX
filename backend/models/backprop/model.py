import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1):
        self.lr = lr
        np.random.seed(42)
        self.W1 = np.random.uniform(-1, 1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.uniform(-1, 1, size=(hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
        
    def backward(self, X, y):
        m = X.shape[0]
        # Requires forward to be called first to populate a2
        error = y - self.a2
        d_output = error * sigmoid_derivative(self.a2)
        
        error_hidden = np.dot(d_output, self.W2.T)
        d_hidden = error_hidden * sigmoid_derivative(self.a1)
        
        self.W2 += np.dot(self.a1.T, d_output) * self.lr / m
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.lr / m
        self.W1 += np.dot(X.T, d_hidden) * self.lr / m
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.lr / m
        
        return np.mean(np.square(error))

    def predict(self, X):
        preds = self.forward(X)
        return np.where(preds >= 0.5, 1, 0)
